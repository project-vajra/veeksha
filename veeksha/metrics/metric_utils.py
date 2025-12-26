from typing import List, Tuple

import numpy as np

TBC_QUANTILE_FOR_THROUGHPUT = 0.99
TARGET_DEADLINE_MISS_RATE_FOR_THROUGHPUT = 0.1


def get_request_level_deadline_miss_rate(
    inter_chunk_times: List[float],
    ttfc_deadline: float,
    tbc_deadline: float,
    should_ignore_first_chunk: bool = False,
) -> Tuple[float, int, int]:
    # calculate the deadline miss rate for a given deadline between chunks
    total_deadlines = 0
    missed_deadlines = 0
    deadline_slack = 0
    curr_missed_deadlines = 0

    for i, inter_chunk_time in enumerate(inter_chunk_times):
        if i == 0:
            if should_ignore_first_chunk:
                continue
            # treat first chunk specially
            if inter_chunk_time <= ttfc_deadline + deadline_slack:
                deadline_slack += ttfc_deadline - inter_chunk_time
                total_deadlines += 1
                continue
            curr_missed_deadlines = (
                1 + (inter_chunk_time - deadline_slack - ttfc_deadline) // tbc_deadline
            )
        else:
            if inter_chunk_time <= tbc_deadline + deadline_slack:
                deadline_slack += tbc_deadline - inter_chunk_time
                total_deadlines += 1
                continue
            curr_missed_deadlines = (inter_chunk_time - deadline_slack) // tbc_deadline
        missed_deadlines += int(curr_missed_deadlines)
        total_deadlines += int(curr_missed_deadlines)
        # reset as we are starting new deadlines for subsequent chunks
        deadline_slack = 0

    if total_deadlines == 0:
        return 0, 0, 0

    return missed_deadlines / total_deadlines, missed_deadlines, total_deadlines


def get_service_level_deadline_miss_rate(
    request_level_inter_chunk_times: List[List[float]],
    ttfc_deadline: List[float],
    tbc_deadline: List[float],
) -> Tuple[float, int, int]:
    service_level_total_deadlines = 0
    service_level_missed_deadlines = 0
    for i, inter_chunk_times in enumerate(request_level_inter_chunk_times):
        missed_deadlines, total_deadlines = get_request_level_deadline_miss_rate(
            inter_chunk_times, ttfc_deadline[i], tbc_deadline[i]
        )[1:]
        service_level_total_deadlines += total_deadlines
        service_level_missed_deadlines += missed_deadlines
    if service_level_total_deadlines == 0:
        return 0, 0, 0
    return (
        service_level_missed_deadlines / service_level_total_deadlines,
        service_level_missed_deadlines,
        service_level_total_deadlines,
    )


def find_min_tbc_deadline_to_meet(
    inter_chunk_times: List[float],
    target_deadline_miss_rate: float,
    ttfc_deadline: float,
    should_ignore_first_chunk: bool = False,
):
    # find the minimum deadline that meets the target miss rate
    deadline = 1e10
    left = 0
    right = 1e10
    mid = 0
    search_granularity = 1e-4
    while right - left > search_granularity:
        mid = (left + right) / 2
        curr_miss_rate, _, _ = get_request_level_deadline_miss_rate(
            inter_chunk_times,
            ttfc_deadline=ttfc_deadline,
            tbc_deadline=mid,
            should_ignore_first_chunk=should_ignore_first_chunk,
        )
        if curr_miss_rate > target_deadline_miss_rate:
            left = mid + search_granularity
        else:
            deadline = mid
            right = mid - search_granularity

    return deadline


def get_deadline_miss_rate_for_target_tbc_values(
    tbc_times: List[List[float]],
    target_tbc_deadline_array: List[float],
    quantile: float = 0.99,
) -> List[float]:
    # no completed requests
    if len(tbc_times) == 0:
        return [0.0 for _ in target_tbc_deadline_array]
    num_requests = len(tbc_times)
    quantile_based_miss_rate = []
    for tbc_deadline in target_tbc_deadline_array:
        deadline_miss_rate = []
        for i in range(num_requests):
            deadline_miss_rate.append(
                get_request_level_deadline_miss_rate(
                    inter_chunk_times=[0] + tbc_times[i],
                    ttfc_deadline=0,
                    tbc_deadline=tbc_deadline,
                    should_ignore_first_chunk=True,
                )[0]
            )
        quantile_based_miss_rate.append(np.quantile(deadline_miss_rate, quantile))
    return quantile_based_miss_rate


def get_throughput_metrics(
    tpot_times: List[float],
    tbc_times: List[List[float]],
) -> Tuple[float, float, float]:
    assert len(tpot_times) == len(tbc_times)
    num_requests = len(tpot_times)
    # no requests have completed
    if num_requests == 0:
        return 0.0, 0.0, 0.0
    mean_tpot = np.mean(tpot_times)
    tpot_based_throughput = float("inf") if mean_tpot == 0 else float(1 / mean_tpot)

    tbc_times_flattened = []
    for tbc_time in tbc_times:
        tbc_times_flattened.extend(tbc_time)

    if len(tbc_times_flattened) == 0:
        return tpot_based_throughput, 0, 0

    p99_tbc = np.quantile(tbc_times_flattened, TBC_QUANTILE_FOR_THROUGHPUT)
    tbc_slo = []
    for i in range(num_requests):
        tbc_slo.append(
            find_min_tbc_deadline_to_meet(
                inter_chunk_times=[0] + tbc_times[i],
                target_deadline_miss_rate=TARGET_DEADLINE_MISS_RATE_FOR_THROUGHPUT,
                ttfc_deadline=0,
                should_ignore_first_chunk=True,
            )
        )
    tbc_slo = np.array(tbc_slo)
    p99_tbc_slo = np.quantile(tbc_slo, TBC_QUANTILE_FOR_THROUGHPUT)

    tbc_based_throughput = float(1 / p99_tbc)
    deadline_based_throughput = float(1 / p99_tbc_slo)

    return tpot_based_throughput, tbc_based_throughput, deadline_based_throughput
