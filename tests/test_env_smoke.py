import pytest

from envs.sumo_env import TrafficSignalEnv


def test_env_reset_returns_expected_state_shape():
    env = TrafficSignalEnv(
        scenario_config={
            "incoming_lanes": ["north_in_0", "south_in_0", "east_in_0", "west_in_0"],
            "phase_count": 2,
            "min_green": 10,
            "max_green": 60,
            "action_delta_seconds": 5,
        },
        reward_config={
            "waiting_weight": 1.0,
            "queue_weight": 0.5,
            "throughput_weight": 0.2,
            "fairness_weight": 0.1,
        },
        sumo_enabled=False,
    )

    state = env.reset()

    assert state.shape == (15,)
