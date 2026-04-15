from __future__ import annotations


def compute_reward(
    total_waiting: float,
    total_queue: float,
    throughput: float,
    directional_waiting_times: list[float],
    waiting_weight: float,
    queue_weight: float,
    throughput_weight: float,
    fairness_weight: float,
) -> float:
    fairness_adjustment = 0.0
    if directional_waiting_times:
        fairness_adjustment = (
            max(directional_waiting_times) - min(directional_waiting_times)
        ) / 2.0

    return (
        -(waiting_weight * total_waiting + queue_weight * total_queue)
        + throughput_weight * throughput
        + fairness_weight * fairness_adjustment
    )
