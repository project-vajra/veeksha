import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.constants.configuration_constants import SCALE_TO_SECONDS
from veeksha.logger import init_logger

logger = init_logger(__name__)


def load_trace(trace_file: str) -> pd.DataFrame:
    """
    Load a trace file from a csv or jsonl file into a pandas DataFrame without any postprocessing.
    """
    if trace_file.endswith(".jsonl"):
        trace_df = pd.read_json(trace_file, lines=True)
    elif trace_file.endswith(".csv"):
        trace_df = pd.read_csv(trace_file)
    else:
        raise ValueError(f"Unsupported trace file format: {trace_file}")

    if trace_df.empty:
        raise ValueError(f"Trace file {trace_file} is empty.")

    return trace_df


def process_request_length_trace(
    trace_df: pd.DataFrame,
    trace_file: str,
    column_map: Optional[Dict[str, str]] = None,
    prefill_scale_factor: float = 1.0,
    decode_scale_factor: float = 1.0,
    max_tokens: int = -1,
) -> pd.DataFrame:
    """
    Postprocess a trace dataframe containing request input_length and output_length:

    - The input_length and output_length are scaled by `prefill_scale_factor` and `decode_scale_factor` respectively.
    - If `max_tokens` is provided, the input_length and output_length are adjusted to ensure the total number of tokens does not exceed `max_tokens`.
    - The input_length and output_length are converted to integers.
    - The input_length and output_length must be > 0; a ValueError is raised otherwise.
    - The input_length and output_length are clipped to be less than or equal to `max_tokens`.
    - Columns are renamed according to `column_map`.

    Args:
        trace_df: DataFrame containing the trace file.
        trace_file: Path to the trace file, only used for logging.
        column_map: A dictionary of column names to rename.
        prefill_scale_factor: Factor to scale the number of prefill tokens. By default no scaling is applied.
        decode_scale_factor: Factor to scale the number of decode tokens. By default no scaling is applied.
        max_tokens: Maximum number of tokens allowed in a request. By default no scaling is applied. If scaling down,
         the prefill and decode tokens are adjusted proportionally.

    Returns:
        Processed trace dataframe.
    """

    new_trace_df = trace_df.copy()

    if column_map is not None:
        for col in column_map.keys():
            if col not in new_trace_df.columns:
                raise ValueError(
                    f"Length trace file does not have column {col}. Available: {list(new_trace_df.columns)}"
                )
        new_trace_df = new_trace_df.rename(columns=column_map, errors="ignore")

    for col in ["input_length", "output_length"]:
        if col not in new_trace_df.columns:
            raise ValueError(
                f"Length trace file must have column '{col}'. Available: {list(new_trace_df.columns)}"
            )

    # scale prefill and decode tokens
    new_trace_df["input_length"] = new_trace_df["input_length"] * prefill_scale_factor
    new_trace_df["output_length"] = new_trace_df["output_length"] * decode_scale_factor

    # make sure all the prefill and decode counts are integers
    new_trace_df["input_length"] = new_trace_df["input_length"].astype(int)
    new_trace_df["output_length"] = new_trace_df["output_length"].astype(int)

    bad = (new_trace_df["input_length"] <= 0) | (new_trace_df["output_length"] <= 0)
    if bad.any():
        bad_idx = new_trace_df.index[bad].tolist()[:5]
        raise ValueError(
            f"{bad.sum()} rows have nonpositive token counts (e.g., indices {bad_idx}). "
            f"Check trace and scale factors."
        )

    # If max_tokens > 0, ensure total tokens do not exceed it and adjust proportionally
    if max_tokens > 0:
        total_tokens = new_trace_df["input_length"] + new_trace_df["output_length"]

        diff_tokens = (total_tokens - max_tokens).clip(lower=0)

        # proportional adjustment
        input_length_ratio = new_trace_df["input_length"] / total_tokens
        output_length_ratio = new_trace_df["output_length"] / total_tokens

        new_trace_df["input_length"] -= (
            np.ceil(diff_tokens * input_length_ratio)
        ).astype(int)
        new_trace_df["output_length"] -= (
            np.ceil(diff_tokens * output_length_ratio)
        ).astype(int)

        # ensure at least one token after adjustment
        new_trace_df["input_length"] = new_trace_df["input_length"].clip(lower=1)
        new_trace_df["output_length"] = new_trace_df["output_length"].clip(lower=1)

        overflow = (
            new_trace_df["input_length"] + new_trace_df["output_length"] > max_tokens
        )
        if overflow.any():
            bad_idx = new_trace_df.index[overflow].tolist()[:5]
            raise ValueError(
                f"Total tokens after clipping must be <= {max_tokens}. Overflow at indices {bad_idx}."
            )
    else:
        # No max cap: just ensure both are at least one
        new_trace_df["input_length"] = new_trace_df["input_length"].clip(lower=1)
        new_trace_df["output_length"] = new_trace_df["output_length"].clip(lower=1)

    if (new_trace_df["input_length"] <= 0).any():
        bad_idx = new_trace_df.index[new_trace_df["input_length"] <= 0].tolist()[:5]
        raise ValueError(
            f"All prefill tokens must be > 0 in length trace file {trace_file}; e.g., indices {bad_idx}."
        )
    if (new_trace_df["output_length"] <= 0).any():
        bad_idx = new_trace_df.index[new_trace_df["output_length"] <= 0].tolist()[:5]
        raise ValueError(
            f"All decode tokens must be > 0 in length trace file {trace_file}; e.g., indices {bad_idx}."
        )

    # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
    pd_ratio = new_trace_df["input_length"] / new_trace_df["output_length"]

    logger.info(
        "Prompt/decode token ratio stats\n%s",
        pd_ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
    )

    return new_trace_df


def process_request_interval_trace(
    trace_df: pd.DataFrame,
    trace_file: str,
    column_map: Optional[Dict[str, str]] = None,
    time_scale_factor: float = 1.0,
    timestamp_unit: str = "ms",
) -> pd.DataFrame:
    """
    Postprocess a trace dataframe containing request timestamps:

    - Timestamps are converted to seconds (canonical time unit) given `timestamp_unit`.
    - `inter_request_time` is created as the time difference between consecutive requests.
      The first interval equals the first (absolute) timestamp value.
    - `inter_request_time` is scaled by `time_scale_factor`.
    - Columns are renamed according to `column_map`.

    Returns:
        Processed trace dataframe.

    Args:
        trace_df: DataFrame containing the trace file.
        trace_file: Path to the trace file, only used for logging.
        column_map: A dictionary of column names to rename.
        time_scale_factor: Factor to scale the time intervals in the trace. By default no scaling is applied.
        timestamp_unit: Unit of the timestamps in the trace file. Is either 'ms' or 's'.
    """

    new_trace_df = trace_df.copy()

    if column_map is not None:
        for col in column_map.keys():
            if col not in new_trace_df.columns:
                raise ValueError(
                    f"Interval trace file does not have column {col}. Available: {list(new_trace_df.columns)}"
                )
        new_trace_df = new_trace_df.rename(columns=column_map, errors="ignore")

    if "timestamp" not in new_trace_df.columns:
        raise ValueError(
            f"Interval trace file must have column 'timestamp'. Available: {list(new_trace_df.columns)}"
        )

    new_trace_df["timestamp"] = pd.to_numeric(
        new_trace_df["timestamp"], errors="coerce"
    )
    if bool(new_trace_df["timestamp"].isna().to_numpy().any()):
        bad_idx = new_trace_df.index[new_trace_df["timestamp"].isna()].tolist()[:5]
        raise ValueError(
            f"Non-numeric timestamps found in interval trace file {trace_file} at indices {bad_idx}."
        )
    if timestamp_unit != "s":
        new_trace_df["timestamp"] = (
            new_trace_df["timestamp"] * SCALE_TO_SECONDS[timestamp_unit]
        )

    # Fail fast if timestamps are not increasing (unordered requests)
    ts_diff = new_trace_df["timestamp"].diff()
    decreasing = ts_diff < 0
    if bool(decreasing.to_numpy().any()):
        bad_positions = decreasing[decreasing].index.tolist()[:5]
        raise ValueError(
            f"Timestamps are not increasing in interval trace file {trace_file}. "
            f"Decreasing at indices {bad_positions}."
        )

    # The interval for the first request is its own timestamp. Subsequent intervals are the time difference
    # between consecutive requests. .diff() creates a NaN for the first row, which we fill with the first
    # timestamp val
    new_trace_df["inter_request_time"] = (
        new_trace_df["timestamp"].diff().fillna(new_trace_df["timestamp"])
    )

    # scale the interval times
    new_trace_df["inter_request_time"] *= time_scale_factor

    return new_trace_df


def generate_random_prompt(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    num_prompt_tokens: int = 1024,
    corpus_lines: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[str, int]:
    """Generate a random prompt with a given number of tokens.
    Args:
        num_prompt_tokens: The number of tokens to generate in the prompt.
        corpus_lines: List of corpus lines to generate the prompt from.
        rng: Optional random number generator for reproducibility. If None, uses global random.
    Returns:
        A random prompt with the given number of tokens.
    """
    assert corpus_lines is not None, "corpus_lines must be provided"

    if rng is None:
        rng = random.Random()

    if num_prompt_tokens < 0:
        raise ValueError("num_prompt_tokens must be >= 0")
    if num_prompt_tokens == 0:
        logger.info(f"Generated random prompt with 0 tokens.")
        return ("", 0)

    remaining_prompt_tokens = num_prompt_tokens

    token_lines = [
        tokenizer.encode(line, add_special_tokens=False) for line in corpus_lines
    ]
    token_lines = [t for t in token_lines if t]
    if not token_lines:
        raise ValueError("All corpus_lines tokenize to zero tokens.")

    prompt_token_ids: List[int] = []
    rng.shuffle(token_lines)  # randomness in the first pass
    idx = 0
    while remaining_prompt_tokens > 0:
        tokens = token_lines[idx]
        take = min(remaining_prompt_tokens, len(tokens))
        if take:
            prompt_token_ids.extend(tokens[:take])
            remaining_prompt_tokens -= take
        idx += 1
        if idx == len(token_lines):
            idx = 0
            rng.shuffle(token_lines)  # reshuffle each full pass

    prompt = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
    return (prompt, num_prompt_tokens)


def generate_random_prompt_fast(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    pretokenized_lines: List[List[int]],
    num_prompt_tokens: int,
    rng: random.Random,
) -> Tuple[str, int]:
    """Generate a random prompt using a pre-tokenized corpus.

    Args:
        tokenizer: Tokenizer used for decoding token IDs back to text.
        pretokenized_lines: Corpus lines tokenized into token ID lists.
        num_prompt_tokens: Number of tokens desired in the resulting prompt.
        rng: Optional random number generator for variety

    Returns:
        Tuple of (prompt_text, num_prompt_tokens). The count equals the requested
        num_prompt_tokens by construction.
    """
    if num_prompt_tokens < 0:
        raise ValueError("num_prompt_tokens must be >= 0")
    if num_prompt_tokens == 0:
        logger.info("Generated random prompt with 0 tokens.")
        return ("", 0)

    token_lines = [t for t in pretokenized_lines if t]
    if not token_lines:
        raise ValueError("All pretokenized_lines are empty.")

    remaining = num_prompt_tokens
    prompt_token_ids: List[int] = []

    # Shuffle a working view once per generation for variety
    indices = list(range(len(token_lines)))
    rng.shuffle(indices)
    idx_cursor = 0

    while remaining > 0:
        tokens = token_lines[indices[idx_cursor]]
        take = min(remaining, len(tokens))
        if take:
            prompt_token_ids.extend(tokens[:take])
            remaining -= take
        idx_cursor += 1
        if idx_cursor == len(indices):
            idx_cursor = 0
            rng.shuffle(indices)

    prompt = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
    return (prompt, num_prompt_tokens)


def generate_random_token_ids_fast(
    pretokenized_lines: List[List[int]],
    num_tokens: int,
    rng: random.Random,
) -> List[int]:
    """Assemble exactly num_tokens token IDs from a pre-tokenized corpus.

    This avoids per-call tokenization and returns token IDs suitable for
    concatenation with other token ID sequences before a single decode.

    Args:
        pretokenized_lines: Corpus lines tokenized into token ID lists.
        num_tokens: Target number of token IDs to assemble.
        rng: Random generator for variety.

    Returns:
        List of token IDs of length exactly num_tokens.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens must be >= 0")
    if num_tokens == 0:
        return []

    token_lines = [t for t in pretokenized_lines if t]
    if not token_lines:
        raise ValueError("All pretokenized_lines are empty.")

    remaining = num_tokens
    out: List[int] = []

    indices = list(range(len(token_lines)))
    rng.shuffle(indices)
    idx_cursor = 0

    while remaining > 0:
        tokens = token_lines[indices[idx_cursor]]
        take = min(remaining, len(tokens))
        if take:
            out.extend(tokens[:take])
            remaining -= take
        idx_cursor += 1
        if idx_cursor == len(indices):
            idx_cursor = 0
            rng.shuffle(indices)

    return out

def base10_to_basen(x, n):
    assert x >= 0
    assert n >= 2
    digits = []
    while x > 0:
        digits.append(x%n)
        x = x // n
    digits.reverse()
    return digits