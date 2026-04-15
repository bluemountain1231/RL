from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ddpg import DDPGAgent
from agents.dqn import DQNAgent
from envs.sumo_env import TrafficSignalEnv
from training.baseline import FixedTimeBaselinePolicy
from utils.config import load_yaml
from utils.logger import append_row
from utils.metrics import EpisodeMetrics
from utils.seeding import set_seed


def run_evaluation_episode(env: TrafficSignalEnv, policy, action_type: str, max_steps: int) -> dict[str, float]:
    metrics = EpisodeMetrics()
    state = env.reset()
    for step_index in range(max_steps):
        if action_type == "baseline":
            green_duration = policy.act(step_index)
            action = 1 if green_duration == env.current_green else (2 if green_duration > env.current_green else 0)
            next_state, reward, done, info = env.step(action, action_type="dqn")
        else:
            action = policy.act(state, eval_mode=True)
            next_state, reward, done, info = env.step(action, action_type=action_type)
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


def aggregate_summaries(summaries: list[dict[str, float]]) -> dict[str, float]:
    keys = summaries[0].keys()
    return {f"mean_{key}": statistics.mean(summary[key] for summary in summaries) for key in keys}


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
    parser.add_argument("--agent", choices=["baseline", "dqn", "ddpg"], required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--agent-config", default="")
    args = parser.parse_args()

    base_config_path = Path(args.base_config).resolve()
    base_config = load_yaml(str(base_config_path))
    scenario_config = load_yaml(resolve_config_path(base_config["scenario"], base_config_path))
    reward_config = load_yaml(resolve_config_path(base_config["reward"], base_config_path))
    set_seed(base_config["seed"])
    env = TrafficSignalEnv(scenario_config=scenario_config, reward_config=reward_config, sumo_enabled=False)

    if args.agent == "baseline":
        policy = FixedTimeBaselinePolicy(schedule=[20, 25])
        action_type = "baseline"
    elif args.agent == "dqn":
        agent_config = load_yaml(str(Path(args.agent_config).resolve()))
        policy = DQNAgent(state_dim=len(env.reset()), action_dim=3, **agent_config)
        policy.load(args.checkpoint)
        action_type = "dqn"
    else:
        agent_config = load_yaml(str(Path(args.agent_config).resolve()))
        policy = DDPGAgent(state_dim=len(env.reset()), action_dim=1, **agent_config)
        policy.load(args.checkpoint)
        action_type = "ddpg"

    summaries = [run_evaluation_episode(env, policy, action_type, base_config["max_episode_steps"]) for _ in range(base_config["eval_episodes"])]
    aggregate = aggregate_summaries(summaries)
    append_row(csv_path=f"{base_config['output_dir']}/csv/eval_{args.agent}.csv", row=aggregate)


if __name__ == "__main__":
    main()
