from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeMetrics:
    rewards: list[float] = field(default_factory=list)
    waiting_times: list[float] = field(default_factory=list)
    queue_lengths: list[float] = field(default_factory=list)
    throughputs: list[float] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)

    def record(self, reward: float, waiting_time: float, queue_length: float, throughput: float, speed: float) -> None:
        self.rewards.append(reward)
        self.waiting_times.append(waiting_time)
        self.queue_lengths.append(queue_length)
        self.throughputs.append(throughput)
        self.speeds.append(speed)

    def summary(self) -> dict[str, float]:
        return {
            "episode_reward": sum(self.rewards),
            "average_waiting_time": sum(self.waiting_times) / len(self.waiting_times),
            "average_queue_length": sum(self.queue_lengths) / len(self.queue_lengths),
            "throughput": sum(self.throughputs),
            "average_speed": sum(self.speeds) / len(self.speeds),
        }
