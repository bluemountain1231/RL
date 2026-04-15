FEATURES_PER_LANE = ("queue_length", "average_speed", "waiting_time")


def state_dimension(lane_count: int, phase_count: int) -> int:
    return lane_count * len(FEATURES_PER_LANE) + phase_count + 1
