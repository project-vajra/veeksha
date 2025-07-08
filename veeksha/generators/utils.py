import numpy as np
import pandas as pd

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
    prefill_scale_factor: float = 1.0,
    decode_scale_factor: float = 1.0,
    max_tokens: int = -1,
) -> None:
    """
    Postprocess a trace dataframe containing request input_length and output_length:

    - The input_length and output_length are scaled by `prefill_scale_factor` and `decode_scale_factor` respectively.
    - If `max_tokens` is provided, the input_length and output_length are adjusted to ensure the total number of tokens does not exceed `max_tokens`.
    - The input_length and output_length are converted to integers.
    - The input_length and output_length are clipped to be greater than 0.
    - The input_length and output_length are clipped to be less than or equal to `max_tokens`.

    Trace dataframe is modified in place.

    Args:
        trace_df: DataFrame containing the trace file.
        trace_file: Path to the trace file, only used for logging.
        prefill_scale_factor: Factor to scale the number of prefill tokens. By default no scaling is applied.
        decode_scale_factor: Factor to scale the number of decode tokens. By default no scaling is applied.
        max_tokens: Maximum number of tokens allowed in a request. By default no scaling is applied. If scaling down,
         the prefill and decode tokens are adjusted proportionally.
    """

    for col in ["input_length", "output_length"]:
        if col not in trace_df.columns:
            raise ValueError(f"Trace file must have column '{col}'")

    # scale prefill and decode tokens
    trace_df["input_length"] = trace_df["input_length"] * prefill_scale_factor
    trace_df["output_length"] = trace_df["output_length"] * decode_scale_factor

    # make sure all the prefill and decode counts are integers
    trace_df["input_length"] = trace_df["input_length"].astype(int)
    trace_df["output_length"] = trace_df["output_length"].astype(int)

    if max_tokens != -1:
        # make sure the total does not exceed the max tokens, adjust the prefill tokens if needed
        total_tokens = trace_df["input_length"] + trace_df["output_length"]
        diff_tokens = total_tokens - max_tokens
        diff_tokens = diff_tokens.clip(lower=0)

        # deduct the diff tokens from the prefill and decode tokens proportionally
        input_length_ratio = trace_df["input_length"] / total_tokens
        output_length_ratio = trace_df["output_length"] / total_tokens

        trace_df["input_length"] -= (np.ceil(diff_tokens * input_length_ratio)).astype(
            int
        )

        trace_df["output_length"] -= (
            np.ceil(diff_tokens * output_length_ratio)
        ).astype(int)

        # make sure that there is at least one prefill and decode token
        trace_df["input_length"] = trace_df["input_length"].clip(lower=1)
        trace_df["output_length"] = trace_df["output_length"].clip(lower=1)

        assert all(
            trace_df["input_length"] + trace_df["output_length"] <= max_tokens
        ), f"Total tokens after clipping must be less than or equal to {max_tokens}"

    assert all(
        trace_df["input_length"] > 0
    ), f"All prefill tokens in trace file {trace_file} must be greater than 0"

    assert all(
        trace_df["output_length"] > 0
    ), f"All decode tokens in trace file {trace_file} must be greater than 0"

    # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
    pd_ratio = trace_df["input_length"] / trace_df["output_length"]

    logger.info(
        f"Prompt/decode token ratio stats\n:{pd_ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}"
    )

    return None


def process_request_interval_trace(
    trace_df: pd.DataFrame,
    trace_file: str,
    time_scale_factor: float = 1.0,
    ms_to_s: bool = True,
) -> None:
    """
    Postprocess a trace dataframe containing request timestamps `timestamp`:

    - If `ms_to_s` is True, the timestamps are converted to seconds.
    - `inter_request_time` is created as the time difference between consecutive requests.
    - `inter_request_time` is scaled by `time_scale_factor`.

    Trace dataframe is modified in place.

    Args:
        trace_df: DataFrame containing the trace file.
        trace_file: Path to the trace file, only used for logging.
        time_scale_factor: Factor to scale the time intervals in the trace. By default no scaling is applied.
        ms_to_s: If True, the timestamps are converted to seconds.
    """

    if "timestamp" not in trace_df.columns:
        raise ValueError(f"Trace file '{trace_file}' must have column 'timestamp' (ms)")

    if ms_to_s:
        trace_df["timestamp"] = trace_df["timestamp"] / 1000.0

    # The interval for the first request is its own timestamp. Subsequent intervals are the time difference
    # between consecutive requests. .diff() creates a NaN for the first row, which we fill with the first
    # timestamp val
    trace_df["inter_request_time"] = (
        trace_df["timestamp"].diff().fillna(trace_df["timestamp"])
    )

    # scale the interval times
    trace_df["inter_request_time"] *= time_scale_factor

    return None
