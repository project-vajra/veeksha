import json
import os
from typing import List

from veeksha.config.deadline import DeadlineReportConfig
from veeksha.metrics.metric_utils import (
    find_min_tbt_deadline_to_meet,
    get_request_level_deadline_miss_rate,
)
from veeksha.metrics.request_metrics import RequestMetrics


class RequestLevelMetrics:
    """
    Array of metrics for all requests
    """

    def __init__(
        self,
        deadline_config: DeadlineReportConfig,
    ) -> None:
        self.ttft_deadline: float = deadline_config.ttft_deadline
        self.tbt_deadline: float = deadline_config.tbt_deadline
        self.target_deadline_miss_rate: float = (
            deadline_config.target_deadline_miss_rate
        )
        self.request_dispatched_at: List[float] = []
        # audit arrays
        self.planned_dispatch_time_monotonic: List[float] = []
        self.actual_dispatch_time_monotonic: List[float] = []
        self.dispatch_delta_s: List[float] = []
        self.scheduling_type: List[str] = []
        self.stream_first_chunk_monotonic: List[float] = []
        self.stream_last_chunk_monotonic: List[float] = []
        self.client_processing_overhead_s: List[float] = []
        self.dispatch_clock_zero_monotonic: List[float] = []
        self.stream_elapsed_s: List[float] = []
        self.measurement_gap_s: List[float] = []
        self.num_prompt_tokens: List[int] = []
        self.num_output_tokens: List[int] = []
        self.num_total_tokens: List[int] = []
        self.tpot: List[float] = []
        self.ttft: List[float] = []
        self.tbt: List[List[float]] = []
        self.end_to_end_latency: List[float] = []
        self.normalized_end_to_end_latency: List[float] = []
        self.output_throughput: List[float] = []
        self.deadline_miss_rate: List[float] = []
        self.min_tbt_deadline_to_meet: List[float] = []

    def put(self, request_metrics: RequestMetrics):
        # Enforce presence of critical audit fields
        if request_metrics.planned_dispatch_time_monotonic is None:
            raise ValueError(
                f"Missing planned_dispatch_time_monotonic for request_id={request_metrics.request_id}"
            )
        if request_metrics.actual_dispatch_time_monotonic is None:
            raise ValueError(
                f"Missing actual_dispatch_time_monotonic for request_id={request_metrics.request_id}"
            )
        if not request_metrics.scheduling_type:
            raise ValueError(
                f"Missing scheduling_type for request_id={request_metrics.request_id}"
            )
        self.request_dispatched_at.append(request_metrics.request_dispatched_at)
        # audit values (store best-effort; default to 0/empty if None)
        self.planned_dispatch_time_monotonic.append(
            request_metrics.planned_dispatch_time_monotonic
            if request_metrics.planned_dispatch_time_monotonic is not None
            else 0.0
        )
        self.actual_dispatch_time_monotonic.append(
            request_metrics.actual_dispatch_time_monotonic
            if request_metrics.actual_dispatch_time_monotonic is not None
            else 0.0
        )
        if (
            request_metrics.actual_dispatch_time_monotonic is not None
            and request_metrics.planned_dispatch_time_monotonic is not None
        ):
            self.dispatch_delta_s.append(
                request_metrics.actual_dispatch_time_monotonic
                - request_metrics.planned_dispatch_time_monotonic
            )
        else:
            self.dispatch_delta_s.append(0.0)
        self.scheduling_type.append(request_metrics.scheduling_type or "")
        self.stream_first_chunk_monotonic.append(
            request_metrics.stream_first_chunk_monotonic or 0.0
        )
        self.stream_last_chunk_monotonic.append(
            request_metrics.stream_last_chunk_monotonic or 0.0
        )
        self.client_processing_overhead_s.append(
            request_metrics.client_processing_overhead_s or 0.0
        )
        self.dispatch_clock_zero_monotonic.append(
            request_metrics.dispatch_clock_zero_monotonic or 0.0
        )
        self.stream_elapsed_s.append(getattr(request_metrics, "stream_elapsed_s", 0.0))
        self.measurement_gap_s.append(
            getattr(request_metrics, "measurement_gap_s", 0.0)
        )
        self.num_prompt_tokens.append(request_metrics.num_prompt_tokens)
        self.num_output_tokens.append(request_metrics.num_output_tokens)
        self.num_total_tokens.append(request_metrics.num_total_tokens)
        self.tpot.append(request_metrics.tpot)
        self.ttft.append(request_metrics.ttft)
        self.tbt.append(request_metrics.inter_token_times[1:])
        self.end_to_end_latency.append(request_metrics.end_to_end_latency)
        self.normalized_end_to_end_latency.append(
            request_metrics.normalized_end_to_end_latency
        )
        self.output_throughput.append(request_metrics.output_throughput)

        ttft_deadline = self.ttft_deadline

        deadline_miss_rate, _, _ = get_request_level_deadline_miss_rate(
            inter_token_times=request_metrics.inter_token_times,
            ttft_deadline=ttft_deadline,
            tbt_deadline=self.tbt_deadline,
        )
        self.deadline_miss_rate.append(deadline_miss_rate)
        min_tbt_deadline_to_meet = find_min_tbt_deadline_to_meet(
            inter_token_times=request_metrics.inter_token_times,
            ttft_deadline=ttft_deadline,
            target_deadline_miss_rate=self.target_deadline_miss_rate,
        )
        self.min_tbt_deadline_to_meet.append(min_tbt_deadline_to_meet)

    def put_dispatch_only(self, request_metrics: RequestMetrics):
        """Record only dispatch time for errored requests."""
        if request_metrics.planned_dispatch_time_monotonic is None:
            raise ValueError(
                f"Missing planned_dispatch_time_monotonic for request_id={request_metrics.request_id} (errored)"
            )
        if request_metrics.actual_dispatch_time_monotonic is None:
            raise ValueError(
                f"Missing actual_dispatch_time_monotonic for request_id={request_metrics.request_id} (errored)"
            )
        if not request_metrics.scheduling_type:
            raise ValueError(
                f"Missing scheduling_type for request_id={request_metrics.request_id} (errored)"
            )
        self.request_dispatched_at.append(request_metrics.request_dispatched_at)
        self.planned_dispatch_time_monotonic.append(
            request_metrics.planned_dispatch_time_monotonic or 0.0
        )
        self.actual_dispatch_time_monotonic.append(
            request_metrics.actual_dispatch_time_monotonic or 0.0
        )
        if (
            request_metrics.actual_dispatch_time_monotonic is not None
            and request_metrics.planned_dispatch_time_monotonic is not None
        ):
            self.dispatch_delta_s.append(
                request_metrics.actual_dispatch_time_monotonic
                - request_metrics.planned_dispatch_time_monotonic
            )
        else:
            self.dispatch_delta_s.append(0.0)
        self.scheduling_type.append(request_metrics.scheduling_type or "")
        self.stream_first_chunk_monotonic.append(
            request_metrics.stream_first_chunk_monotonic or 0.0
        )
        self.stream_last_chunk_monotonic.append(
            request_metrics.stream_last_chunk_monotonic or 0.0
        )
        self.client_processing_overhead_s.append(
            request_metrics.client_processing_overhead_s or 0.0
        )
        self.dispatch_clock_zero_monotonic.append(
            request_metrics.dispatch_clock_zero_monotonic or 0.0
        )
        self.stream_elapsed_s.append(getattr(request_metrics, "stream_elapsed_s", 0.0))
        self.measurement_gap_s.append(
            getattr(request_metrics, "measurement_gap_s", 0.0)
        )

    def to_dict(self):
        return {
            "request_dispatched_at": self.request_dispatched_at,
            "planned_dispatch_time_monotonic": self.planned_dispatch_time_monotonic,
            "actual_dispatch_time_monotonic": self.actual_dispatch_time_monotonic,
            "dispatch_delta_s": self.dispatch_delta_s,
            "scheduling_type": self.scheduling_type,
            "stream_first_chunk_monotonic": self.stream_first_chunk_monotonic,
            "stream_last_chunk_monotonic": self.stream_last_chunk_monotonic,
            "client_processing_overhead_s": self.client_processing_overhead_s,
            "dispatch_clock_zero_monotonic": self.dispatch_clock_zero_monotonic,
            "stream_elapsed_s": self.stream_elapsed_s,
            "measurement_gap_s": self.measurement_gap_s,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_output_tokens": self.num_output_tokens,
            "num_total_tokens": self.num_total_tokens,
            "tpot": self.tpot,
            "ttft": self.ttft,
            "tbt": self.tbt,
            "end_to_end_latency": self.end_to_end_latency,
            "normalized_end_to_end_latency": self.normalized_end_to_end_latency,
            "output_throughput": self.output_throughput,
            "deadline_miss_rate": self.deadline_miss_rate,
            "min_tbt_deadline_to_meet": self.min_tbt_deadline_to_meet,
        }

    def save(self, output_dir: str):
        with open(os.path.join(output_dir, "request_level_metrics.json"), "w") as f:
            json.dump(self.to_dict(), f)
