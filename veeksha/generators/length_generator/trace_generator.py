import ast
from typing import Tuple

import numpy as np
import pandas as pd

from veeksha.config.config import TraceRequestLengthGeneratorConfig
from veeksha.logger import init_logger
from veeksha.generators.length_generator.base_generator import (
    BaseRequestLengthGenerator,
)
from typing import Dict, Union

logger = init_logger(__name__)


class TraceRequestLengthGenerator(BaseRequestLengthGenerator):
    def __init__(self, config: TraceRequestLengthGeneratorConfig):
        self.config = config

        logger.info(f"Loading trace file {self.config.trace_file}")
        trace_file = self.config.trace_file
        if trace_file.endswith(".jsonl"):
            self.trace_df = pd.read_json(trace_file, lines=True)
        elif trace_file.endswith(".csv"):
            self.trace_df = pd.read_csv(trace_file)
        else:
            raise ValueError(f"Unsupported trace file format: {trace_file}")

        for col in ["input_length", "output_length"]:
            if col not in self.trace_df.columns:
                raise ValueError(f"Trace file '{trace_file}' must have column '{col}'")

        self._has_hash_ids = "hash_ids" in self.trace_df.columns

        if self._has_hash_ids:
            if not isinstance(self.trace_df["hash_ids"].iloc[0], list):
                self.trace_df["hash_ids"] = self.trace_df["hash_ids"].apply(
                    ast.literal_eval
                )

        # scale prefill and decode tokens
        self.trace_df["input_length"] = (
            self.trace_df["input_length"] * self.config.prefill_scale_factor
        )
        self.trace_df["output_length"] = (
            self.trace_df["output_length"] * self.config.decode_scale_factor
        )

        # make sure all the prefill and decode counts are integers
        self.trace_df["input_length"] = self.trace_df[
            "input_length"
        ].astype(int)
        self.trace_df["output_length"] = self.trace_df["output_length"].astype(
            int
        )

        # make sure the total does not exceed the max tokens, adjust the prefill tokens if needed
        total_tokens = (
            self.trace_df["input_length"] + self.trace_df["output_length"]
        )
        diff_tokens = total_tokens - self.config.max_tokens
        diff_tokens = diff_tokens.clip(lower=0)

        # dedcut the diff tokens from the prefill and decode tokens proportionally
        input_length_ratio = self.trace_df["input_length"] / total_tokens
        output_length_ratio = self.trace_df["output_length"] / total_tokens

        self.trace_df["input_length"] -= (
            np.ceil(diff_tokens * input_length_ratio)
        ).astype(int)

        self.trace_df["output_length"] -= (
            np.ceil(diff_tokens * output_length_ratio)
        ).astype(int)

        # make sure that there is at least one prefill and decode token
        self.trace_df["input_length"] = self.trace_df["input_length"].clip(
            lower=1
        )
        self.trace_df["output_length"] = self.trace_df["output_length"].clip(
            lower=1
        )

        assert all(
            self.trace_df["input_length"] + self.trace_df["output_length"]
            <= self.config.max_tokens
        )

        assert all(self.trace_df["input_length"] > 0)

        assert all(self.trace_df["output_length"] > 0)

        # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
        pd_ratio = (
            self.trace_df["input_length"] / self.trace_df["output_length"]
        )
        logger.info(
            f"Loaded request length trace file {trace_file} with {len(self.trace_df)} requests"
        )
        logger.info(
            f"Prompt/decode token ratio stats\n:{pd_ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}"
        )

        self.next_request_idx = 0

    def get_next_num_tokens(self) -> Tuple[float, float]:
        if self.next_request_idx >= len(self.trace_df):
            return -1, -1

        row = self.trace_df.iloc[self.next_request_idx]
        self.next_request_idx += 1

        return (
            row["input_length"],
            row["output_length"],
        )

    def get_next_request_params(self) -> Dict[str, Union[int, float]]:
        assert self._has_hash_ids
        if self.next_request_idx >= len(self.trace_df):
            return [], -1, -1, -1, -1, -1, -1

        row = self.trace_df.iloc[self.next_request_idx]
        self.next_request_idx += 1

        hash_ids = row["hash_ids"]
        hash_count = len(hash_ids)
        input_length = row["input_length"]
        output_length = row["output_length"]
        block_count = (
            input_length + self.config.block_size - 1
        ) // self.config.block_size

        # todo sessions
        request_id = row["request_id"]
        session_id = row["session_id"]
        num_requests_in_session = row["num_requests_in_session"]
        prefix_match_pct = row['prefix_match_pct']

        assert hash_count >= block_count, f"{hash_count} >= {block_count}"

        return (hash_ids, int(input_length), int(output_length), request_id, session_id, num_requests_in_session, prefix_match_pct)

    def has_hash_ids(self) -> bool:
        return self._has_hash_ids

    def get_block_size(self) -> int:
        assert self.has_hash_ids()
        return self.config.block_size
