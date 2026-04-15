from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from envs.action import apply_green_constraints, decode_ddpg_action, decode_dqn_action
from envs.state_builder import build_state
from envs.reward import compute_reward


@dataclass
class LaneSnapshot:
    queue_length: float
    average_speed: float
    waiting_time: float


class TrafficSignalEnv:
    def __init__(self, scenario_config: dict, reward_config: dict, sumo_enabled: bool = True) -> None:
        self.scenario_config = scenario_config
        self.reward_config = reward_config
        self.sumo_enabled = sumo_enabled
        self.incoming_lanes = scenario_config["incoming_lanes"]
        self.phase_count = scenario_config["phase_count"]
        self.current_phase = 0
        self.phase_elapsed = 0.0
        self.current_green = scenario_config["min_green"]
        self.simulation_step = 0
        self.traci = None

    def reset(self) -> np.ndarray:
        self.current_phase = 0
        self.phase_elapsed = 0.0
        self.current_green = self.scenario_config["min_green"]
        self.simulation_step = 0
        return self._build_observation(self._mock_lane_metrics())

    def step(self, action: int | float, action_type: str = "dqn") -> tuple[np.ndarray, float, bool, dict]:
        if action_type == "dqn":
            delta = decode_dqn_action(int(action), self.scenario_config["action_delta_seconds"])
        else:
            delta = decode_ddpg_action(float(action), self.scenario_config["action_delta_seconds"])

        self.current_green = apply_green_constraints(
            proposed_green=self.current_green + delta,
            min_green=self.scenario_config["min_green"],
            max_green=self.scenario_config["max_green"],
        )
        self.phase_elapsed += 1.0
        self.simulation_step += 1
        lane_metrics = self._mock_lane_metrics()
        observation = self._build_observation(lane_metrics)
        totals = self._totals(lane_metrics)
        reward = compute_reward(
            total_waiting=totals["total_waiting"],
            total_queue=totals["total_queue"],
            throughput=totals["throughput"],
            directional_waiting_times=totals["directional_waiting_times"],
            waiting_weight=self.reward_config["waiting_weight"],
            queue_weight=self.reward_config["queue_weight"],
            throughput_weight=self.reward_config["throughput_weight"],
            fairness_weight=self.reward_config["fairness_weight"],
        )
        done = self.simulation_step >= 20
        info = {
            "average_waiting_time": totals["total_waiting"] / len(self.incoming_lanes),
            "average_queue_length": totals["total_queue"] / len(self.incoming_lanes),
            "throughput": totals["throughput"],
            "average_speed": totals["average_speed"],
            "current_green": self.current_green,
        }
        return observation, reward, done, info

    def _build_observation(self, lane_metrics: dict[str, dict[str, float]]) -> np.ndarray:
        return build_state(
            lane_metrics=lane_metrics,
            lane_order=self.incoming_lanes,
            current_phase=self.current_phase,
            phase_count=self.phase_count,
            phase_elapsed=self.phase_elapsed,
        )

    def _mock_lane_metrics(self) -> dict[str, dict[str, float]]:
        base = max(0.0, 5.0 - 0.1 * self.simulation_step)
        return {
            lane_id: {
                "queue_length": base + index,
                "average_speed": 6.0 + 0.5 * index,
                "waiting_time": 10.0 + index + self.simulation_step,
            }
            for index, lane_id in enumerate(self.incoming_lanes)
        }

    def _totals(self, lane_metrics: dict[str, dict[str, float]]) -> dict[str, float | list[float]]:
        waiting_times = [metrics["waiting_time"] for metrics in lane_metrics.values()]
        queue_lengths = [metrics["queue_length"] for metrics in lane_metrics.values()]
        speeds = [metrics["average_speed"] for metrics in lane_metrics.values()]
        return {
            "total_waiting": sum(waiting_times),
            "total_queue": sum(queue_lengths),
            "throughput": float(self.simulation_step + 1),
            "directional_waiting_times": waiting_times,
            "average_speed": sum(speeds) / len(speeds),
        }
