from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ddpg import DDPGAgent
from agents.dqn import DQNAgent
from training.baseline import FixedTimeBaselinePolicy
from envs.sumo_env import TrafficSignalEnv
from utils.config import load_yaml
from utils.logger import append_row
from utils.metrics import EpisodeMetrics
from utils.plotting import save_line_plot
from utils.seeding import set_seed


def run_training_episode(env: TrafficSignalEnv, agent, action_type: str, max_steps: int) -> dict[str, float]:
    metrics = EpisodeMetrics()
    state = env.reset()
    for _ in range(max_steps):
        action = agent.act(state, eval_mode=False)
        next_state, reward, done, info = env.step(action, action_type=action_type)
        if action_type == "dqn":
            stored_action = np.asarray([action], dtype=np.int64)
        else:
            stored_action = np.asarray([action], dtype=np.float32)
        agent.remember((state, stored_action, reward, next_state, done))
        agent.update()
        metrics.record(
            reward=reward,
            waiting_time=info["average_waiting_time"],
            queue_length=info["average_queue_length"],
            throughput=info["throughput"],
            speed=info["average_speed"],
        )
        state = next_state
        if done:
            break
    return metrics.summary()


def build_agent(agent_name: str, state_dim: int, config: dict):
    if agent_name == "dqn":
        return DQNAgent(state_dim=state_dim, action_dim=3, **config), "dqn"
    if agent_name == "ddpg":
        return DDPGAgent(state_dim=state_dim, action_dim=1, **config), "ddpg"
    raise ValueError(f"Unsupported agent: {agent_name}")


def resolve_config_path(config_path: str, reference_path: Path) -> str:
    path = Path(config_path)
    if path.is_absolute():
        return str(path)
    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return str(project_candidate)
    return str((reference_path.parent / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--agent-config", required=True)
    args = parser.parse_args()

    base_config_path = Path(args.base_config).resolve()
    base_config = load_yaml(str(base_config_path))
    agent_config = load_yaml(str(Path(args.agent_config).resolve()))
    scenario_config = load_yaml(resolve_config_path(base_config["scenario"], base_config_path))
    reward_config = load_yaml(resolve_config_path(base_config["reward"], base_config_path))
    set_seed(base_config["seed"])

    env = TrafficSignalEnv(scenario_config=scenario_config, reward_config=reward_config, sumo_enabled=False)
    state_dim = len(env.reset())
    agent, action_type = build_agent(base_config["agent"], state_dim, agent_config)

    episode_rewards: list[float] = []
    for episode_index in range(base_config["train_episodes"]):
        summary = run_training_episode(env, agent, action_type, base_config["max_episode_steps"])
        episode_rewards.append(summary["episode_reward"])
        append_row(
            csv_path=f"{base_config['output_dir']}/csv/train_{base_config['agent']}.csv",
            row={"episode": episode_index, **summary},
        )
        if (episode_index + 1) % base_config["checkpoint_every"] == 0:
            Path(f"{base_config['output_dir']}/checkpoints").mkdir(parents=True, exist_ok=True)
            agent.save(f"{base_config['output_dir']}/checkpoints/{base_config['agent']}_{episode_index + 1}.pt")

    save_line_plot(
        values=episode_rewards,
        title=f"Training Reward - {base_config['agent'].upper()}",
        y_label="Episode Reward",
        output_path=f"{base_config['output_dir']}/plots/{base_config['agent']}_reward.png",
    )


if __name__ == "__main__":
    main()
