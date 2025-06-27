import pandas as pd

from veeksha.config.config import TraceRequestIntervalGeneratorConfig
from veeksha.logger import init_logger
from veeksha.generators.interval_generator.base_generator import (
    BaseRequestIntervalGenerator,
)

logger = init_logger(__name__)


class TraceRequestIntervalGenerator(BaseRequestIntervalGenerator):
    """
    Reads a trace csv file containing request arrival time, its prompt and completion token values to generate
    inter-request times, number of tokens.
    """

    def __init__(self, config: TraceRequestIntervalGeneratorConfig):
        self.config = config

        trace_file = self.config.trace_file

        if trace_file.endswith(".jsonl"):
            self.trace_df = pd.read_json(trace_file, lines=True)
        elif trace_file.endswith(".csv"):
            self.trace_df = pd.read_csv(trace_file)
        else:
            raise ValueError(f"Unsupported trace file format: {trace_file}")

        if "timestamp" not in self.trace_df.columns:
            raise ValueError(f"Trace file '{trace_file}' must have column 'timestamp' (ms)")

        self.trace_df["timestamp"] = self.trace_df["timestamp"] / 1000.0

        # The interval for the first request is its own timestamp. Subsequent intervals are the time difference
        # between consecutive requests. .diff() creates a NaN for the first row, which we fill with the first
        # timestamp val
        self.trace_df["inter_request_time"] = (
            self.trace_df["timestamp"].diff().fillna(self.trace_df["timestamp"])
        )

        self.next_request_idx = 0

        logger.info(
            f"Loaded interval trace file {trace_file} with {len(self.trace_df)} requests"
        )

    def get_next_inter_request_time(self) -> float:
        if self.next_request_idx >= len(self.trace_df):
            return -1

        inter_request_time = self.trace_df.iloc[self.next_request_idx][
            "inter_request_time"
        ]
        self.next_request_idx += 1
        return inter_request_time
