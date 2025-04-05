from dataclasses import dataclass, field
from functools import cached_property
from statistics import mean
from typing import List, Optional


@dataclass
class RequestMetrics:
    """
    Request-level metrics for 1 request, all metrics are in seconds.
    """

    request_dispatched_at: float = 0.0
    inter_token_times: List[float] = field(default_factory=list)
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0
    error_msg: Optional[str] = None
    error_code: Optional[int] = None

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

    @cached_property
    def token_arrival_times(self):
        """
        Arrival times for decoded tokens
        """
        if not self.inter_token_times or self.num_output_tokens == 0:
            return []
            
        arrival_times = []
        cumulative_time = self.request_dispatched_at + self.inter_token_times[0]
        
        for t in self.inter_token_times[1:]:
            cumulative_time += t
            arrival_times.append(cumulative_time)
                
        return arrival_times
