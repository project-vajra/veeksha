from typing import Dict, Optional

import numpy as np
import pandas as pd

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
        if np.any(total_tokens.to_numpy() == 0):
            raise ValueError("Zero total tokens after scaling; cannot compute ratios.")

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
                f"Total tokens after clipping must be less <= {max_tokens}. Overflow at indices {bad_idx}."
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
