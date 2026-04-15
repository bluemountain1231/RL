import numpy as np

from envs.state_builder import build_state


def test_build_state_orders_features_by_lane_then_phase():
    lane_metrics = {
        "north_in_0": {"queue_length": 2.0, "average_speed": 8.0, "waiting_time": 12.0},
        "south_in_0": {"queue_length": 1.0, "average_speed": 7.5, "waiting_time": 9.0},
        "east_in_0": {"queue_length": 3.0, "average_speed": 6.0, "waiting_time": 15.0},
        "west_in_0": {"queue_length": 0.0, "average_speed": 9.5, "waiting_time": 4.0},
    }

    state = build_state(
        lane_metrics=lane_metrics,
        lane_order=["north_in_0", "south_in_0", "east_in_0", "west_in_0"],
        current_phase=1,
        phase_count=2,
        phase_elapsed=6.0,
    )

    expected = np.array([
        2.0, 8.0, 12.0,
        1.0, 7.5, 9.0,
        3.0, 6.0, 15.0,
        0.0, 9.5, 4.0,
        0.0, 1.0,
        6.0,
    ], dtype=np.float32)

    assert np.allclose(state, expected)
