import numpy as np

from agents.ddpg import DDPGAgent
from agents.dqn import DQNAgent

from training.baseline import FixedTimeBaselinePolicy
from utils.metrics import EpisodeMetrics


def test_fixed_time_baseline_cycles_green_duration_values():
    policy = FixedTimeBaselinePolicy(schedule=[20, 25])
    assert policy.act(step_index=0) == 20
    assert policy.act(step_index=1) == 25
    assert policy.act(step_index=2) == 20


def test_fixed_time_baseline_rejects_empty_schedule():
    try:
        FixedTimeBaselinePolicy(schedule=[])
        assert False, "Expected ValueError for empty schedule"
    except ValueError as exc:
        assert "empty" in str(exc).lower()


def test_episode_metrics_summary_returns_zeroes_for_empty_episode():
    metrics = EpisodeMetrics()

    assert metrics.summary() == {
        "episode_reward": 0.0,
        "average_waiting_time": 0.0,
        "average_queue_length": 0.0,
        "throughput": 0.0,
        "average_speed": 0.0,
    }


def test_dqn_agent_returns_valid_action_index():
    agent = DQNAgent(
        state_dim=15,
        action_dim=3,
        hidden_dim=32,
        gamma=0.99,
        learning_rate=0.001,
        batch_size=2,
        target_update_freq=5,
    )
    action = agent.act(np.zeros(15, dtype=np.float32), eval_mode=True)
    assert action in {0, 1, 2}


def test_ddpg_agent_returns_scalar_action():
    agent = DDPGAgent(
        state_dim=15,
        action_dim=1,
        hidden_dim=32,
        gamma=0.99,
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
        batch_size=2,
        tau=0.005,
        noise_std=0.1,
    )
    action = agent.act(np.zeros(15, dtype=np.float32), eval_mode=True)
    assert isinstance(action, float)
    assert -1.0 <= action <= 1.0
