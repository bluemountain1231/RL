import numpy as np
import torch

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


def test_ddpg_agent_returns_vector_action_for_multi_dimensional_control():
    agent = DDPGAgent(
        state_dim=15,
        action_dim=2,
        hidden_dim=32,
        gamma=0.99,
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
        batch_size=2,
        tau=0.005,
        noise_std=0.1,
    )
    with torch.no_grad():
        final_linear = agent.actor.model[4]
        final_linear.weight.zero_()
        final_linear.bias.copy_(torch.tensor([2.0, -2.0]))

    action = agent.act(np.zeros(15, dtype=np.float32), eval_mode=True)

    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)


def test_ddpg_agent_update_accepts_scalar_continuous_actions_from_replay():
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
    for reward in (1.0, 0.5):
        agent.remember(
            (
                np.zeros(15, dtype=np.float32),
                0.25,
                reward,
                np.ones(15, dtype=np.float32),
                False,
            )
        )

    loss = agent.update()

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_dqn_agent_save_and_load_restore_training_state(tmp_path):
    agent = DQNAgent(
        state_dim=15,
        action_dim=3,
        hidden_dim=32,
        gamma=0.99,
        learning_rate=0.001,
        batch_size=2,
        target_update_freq=1,
        epsilon_start=0.9,
        epsilon_end=0.1,
        epsilon_decay=0.5,
    )
    for index in range(2):
        agent.remember(
            (
                np.full(15, index, dtype=np.float32),
                index % 3,
                1.0,
                np.full(15, index + 1, dtype=np.float32),
                False,
            )
        )
    agent.update()
    checkpoint_path = tmp_path / "dqn.pt"

    agent.save(str(checkpoint_path))

    restored = DQNAgent(
        state_dim=15,
        action_dim=3,
        hidden_dim=32,
        gamma=0.99,
        learning_rate=0.001,
        batch_size=2,
        target_update_freq=1,
        epsilon_start=0.2,
        epsilon_end=0.1,
        epsilon_decay=0.9,
    )
    restored.load(str(checkpoint_path))

    assert restored.epsilon == agent.epsilon
    assert restored.update_steps == agent.update_steps
    assert restored.optimizer.state_dict()["state"]
    assert restored.optimizer.state_dict()["state"].keys() == agent.optimizer.state_dict()["state"].keys()
    for restored_param, saved_param in zip(restored.online_net.parameters(), agent.online_net.parameters(), strict=True):
        assert torch.allclose(restored_param, saved_param)
    for restored_param, saved_param in zip(restored.target_net.parameters(), agent.target_net.parameters(), strict=True):
        assert torch.allclose(restored_param, saved_param)


def test_ddpg_agent_save_and_load_restore_training_state(tmp_path):
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
    for reward in (1.0, 0.5):
        agent.remember(
            (
                np.zeros(15, dtype=np.float32),
                np.array([0.25], dtype=np.float32),
                reward,
                np.ones(15, dtype=np.float32),
                False,
            )
        )
    agent.update()
    checkpoint_path = tmp_path / "ddpg.pt"

    agent.save(str(checkpoint_path))

    restored = DDPGAgent(
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
    restored.load(str(checkpoint_path))

    assert restored.actor_optimizer.state_dict()["state"]
    assert restored.critic_optimizer.state_dict()["state"]
    for restored_param, saved_param in zip(restored.actor.parameters(), agent.actor.parameters(), strict=True):
        assert torch.allclose(restored_param, saved_param)
    for restored_param, saved_param in zip(restored.critic.parameters(), agent.critic.parameters(), strict=True):
        assert torch.allclose(restored_param, saved_param)
    for restored_param, saved_param in zip(restored.target_actor.parameters(), agent.target_actor.parameters(), strict=True):
        assert torch.allclose(restored_param, saved_param)
    for restored_param, saved_param in zip(restored.target_critic.parameters(), agent.target_critic.parameters(), strict=True):
        assert torch.allclose(restored_param, saved_param)
