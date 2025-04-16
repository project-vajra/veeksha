import pandas as pd

from veeksha.config.config import TraceRequestIntervalGeneratorConfig
from veeksha.logger import init_logger
from veeksha.request_generator.interval_generator.base_generator import (
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
            self.trace_df.rename(columns={"timestamp": "arrival_time"}, inplace=True)
            self.trace_df["arrival_time"] = pd.to_datetime(self.trace_df["arrival_time"], unit="ms")
        elif trace_file.endswith(".csv"):
            self.trace_df = pd.read_csv(trace_file)
            self.trace_df["arrival_time"] = pd.to_datetime(self.trace_df["arrival_time"])
        else:
            raise ValueError(f"Unsupported trace file format: {trace_file}")

        print("Before restricting trace_df:")
        print(self.trace_df)
        # restrict trace_df to be a subset of rows that have the same date
        self.trace_df = self.trace_df[
            (self.trace_df["arrival_time"] > self.config.start_time)
            & (self.trace_df["arrival_time"] < self.config.end_time)
        ]

        print("After restricting trace_df (start, end):", self.config.start_time, self.config.end_time)
        print(self.trace_df)

        # change back to seconds
        self.trace_df["arrival_time"] = (
            self.trace_df["arrival_time"] - self.trace_df["arrival_time"].min()
        ) // pd.Timedelta("1s")

        # rescale the time to change QPS
        # self.trace_df["arrival_time"] = (
        #     self.trace_df["arrival_time"] * self.config.time_scale_factor
        # )

        # compute the inter-request time
        self.trace_df["inter_request_time"] = self.trace_df["arrival_time"].diff()  # type: ignore

        self.next_request_idx = 1

        logger.info(
            f"Loaded interval trace file {trace_file} with {len(self.trace_df)} requests"
        )

        logger.info(self.trace_df)

    def get_next_inter_request_time(self) -> float:
        if self.next_request_idx >= len(self.trace_df):
            return -1

        inter_request_time = self.trace_df.iloc[self.next_request_idx][
            "inter_request_time"
        ]
        self.next_request_idx += 1
        return inter_request_time
