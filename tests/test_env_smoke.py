import pytest

from agents.dqn import DQNAgent
from envs.sumo_env import TrafficSignalEnv
from training.evaluate import run_evaluation_episode
from training.baseline import FixedTimeBaselinePolicy
from training.train import run_training_episode


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


def test_mock_env_step_returns_expected_contract():
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

    env.reset()
    observation, reward, done, info = env.step(action=1, action_type="dqn")

    assert observation.shape == (15,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert list(info.keys()) == [
        "average_waiting_time",
        "average_queue_length",
        "throughput",
        "average_speed",
        "current_green",
    ]


def test_mock_env_step_invalid_action_type_raises_value_error():
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

    env.reset()

    with pytest.raises(ValueError, match="Unsupported action_type: invalid"):
        env.step(action=1, action_type="invalid")


def test_mock_env_step_does_not_end_episode_implicitly():
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

    env.reset()
    done = False
    for _ in range(25):
        _, _, done, _ = env.step(action=1, action_type="dqn")

    assert done is False


def test_run_training_episode_returns_summary_for_dqn():
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
    agent = DQNAgent(
        state_dim=15,
        action_dim=3,
        hidden_dim=32,
        gamma=0.99,
        learning_rate=0.001,
        batch_size=2,
        target_update_freq=5,
    )

    summary = run_training_episode(env=env, agent=agent, action_type="dqn", max_steps=5)

    assert set(summary.keys()) == {
        "episode_reward",
        "average_waiting_time",
        "average_queue_length",
        "throughput",
        "average_speed",
    }


def test_run_evaluation_episode_supports_baseline_policy():
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
    policy = FixedTimeBaselinePolicy(schedule=[20, 25])

    summary = run_evaluation_episode(env=env, policy=policy, action_type="baseline", max_steps=5)

    assert set(summary.keys()) == {
        "episode_reward",
        "average_waiting_time",
        "average_queue_length",
        "throughput",
        "average_speed",
    }
