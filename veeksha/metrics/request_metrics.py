from dataclasses import dataclass
from functools import cached_property
from statistics import mean
from typing import List, Optional


@dataclass
class RequestMetrics:
    """
    Request-level metrics for 1 request, all metrics are in seconds.
    """

    request_dispatched_at: float
    inter_token_times: List[float]
    num_prompt_tokens: int
    num_output_tokens: int
    error_msg: Optional[str] = None
    error_code: Optional[int] = None
    # Request id for correlation when Response is None
    request_id: Optional[int] = None
    # Dispatch/timing audit fields (absolute monotonic seconds where applicable)
    planned_dispatch_time_monotonic: Optional[float] = None
    actual_dispatch_time_monotonic: Optional[float] = None
    scheduling_type: Optional[str] = None  # "session" | "non_session"
    # Streaming timing audit
    stream_first_chunk_monotonic: Optional[float] = None
    stream_last_chunk_monotonic: Optional[float] = None
    client_processing_overhead_s: Optional[float] = None

    @cached_property
    def dispatch_delta_s(self) -> float:
        if (
            self.actual_dispatch_time_monotonic is None
            or self.planned_dispatch_time_monotonic is None
        ):
            return 0.0
        return self.actual_dispatch_time_monotonic - self.planned_dispatch_time_monotonic

    @cached_property
    def stream_elapsed_s(self) -> float:
        if (
            self.stream_first_chunk_monotonic is None
            or self.stream_last_chunk_monotonic is None
        ):
            return 0.0
        return self.stream_last_chunk_monotonic - self.stream_first_chunk_monotonic

    @cached_property
    def measurement_gap_s(self) -> float:
        # Difference between total observed stream span and sum of inter-token times
        elapsed = self.stream_elapsed_s
        if elapsed <= 0.0:
            return 0.0
        gap = elapsed - self.end_to_end_latency
        return gap if gap > 0 else 0.0

    @cached_property
    def num_total_tokens(self):
        return self.num_prompt_tokens + self.num_output_tokens

    @cached_property
    def end_to_end_latency(self):
        return sum(self.inter_token_times)

    @cached_property
    def normalized_end_to_end_latency(self):
        if self.num_output_tokens == 0:
            return 0

        return self.end_to_end_latency / self.num_output_tokens

    @cached_property
    def ttft(self):
        if self.num_output_tokens == 0:
            return 0

        # non-streaming responses have no token timing data
        if not self.inter_token_times:
            return 0

        return self.inter_token_times[0]

    @cached_property
    def tpot(self):
        if self.num_output_tokens == 0:
            return 0

        if len(self.inter_token_times) < 2:
            return 0

        return mean(self.inter_token_times[1:])

    @cached_property
    def output_throughput(self):
        if self.end_to_end_latency == 0:
            return 0

        return self.num_output_tokens / self.end_to_end_latency
