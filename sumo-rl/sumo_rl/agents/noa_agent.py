"""Nutcracker-inspired online phase control agent."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np


class NOAAgent:
    """Nutcracker-inspired online optimizer for discrete traffic-signal phases."""

    def __init__(
        self,
        action_space,
        population_size: int = 12,
        elite_size: int = 4,
        cache_size: int = 256,
        alpha: float = 0.6,
        beta: float = 0.3,
        sigma: float = 0.15,
        recovery_window: int = 6,
        recovery_queue_threshold: float = 0.75,
        recovery_delay_threshold: float = 0.7,
        cache_weight: float = 0.35,
        switch_penalty: float = 0.08,
        weight_queue: float = 0.35,
        weight_delay: float = 0.3,
        weight_pressure: float = 0.15,
        weight_throughput: float = 0.2,
        fitness_weight_queue: float = 0.4,
        fitness_weight_delay: float = 0.3,
        fitness_weight_speed: float = 0.15,
        fitness_weight_throughput: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.action_space = action_space
        self.population_size = max(3, population_size)
        self.elite_size = max(1, min(elite_size, self.population_size))
        self.cache_size = max(1, cache_size)
        self.alpha = alpha
        self.beta = beta
        self.base_sigma = sigma
        self.recovery_window = max(1, recovery_window)
        self.recovery_queue_threshold = recovery_queue_threshold
        self.recovery_delay_threshold = recovery_delay_threshold
        self.base_cache_weight = np.clip(cache_weight, 0.0, 1.0)
        self.switch_penalty = max(0.0, switch_penalty)
        self.weight_queue = weight_queue
        self.weight_delay = weight_delay
        self.weight_pressure = weight_pressure
        self.weight_throughput = weight_throughput
        self.fitness_weight_queue = fitness_weight_queue
        self.fitness_weight_delay = fitness_weight_delay
        self.fitness_weight_speed = fitness_weight_speed
        self.fitness_weight_throughput = fitness_weight_throughput
        self.rng = np.random.default_rng(seed)

        self.num_actions = action_space.n
        self.population = np.zeros((self.population_size, self.num_actions), dtype=np.float32)
        self.elite_population = np.zeros((self.elite_size, self.num_actions), dtype=np.float32)
        self.elite_scores = np.full(self.elite_size, -np.inf, dtype=np.float32)
        self.cache: OrderedDict[tuple[int, ...], dict[str, np.ndarray | float | int]] = OrderedDict()

        self.action: Optional[int] = None
        self.last_scores = np.zeros(self.num_actions, dtype=np.float32)
        self.last_state_key: Optional[tuple[int, ...]] = None
        self.last_observation: Optional[np.ndarray] = None
        self.last_queue = 0.0
        self.last_delay = 0.0
        self.best_fitness = -np.inf
        self.stagnation_count = 0
        self.recovery_steps_left = 0
        self.step_count = 0

    def reset(self) -> None:
        """Reset the agent internal state."""
        self.population.fill(0.0)
        self.elite_population.fill(0.0)
        self.elite_scores.fill(-np.inf)
        self.cache.clear()
        self.action = None
        self.last_scores.fill(0.0)
        self.last_state_key = None
        self.last_observation = None
        self.last_queue = 0.0
        self.last_delay = 0.0
        self.best_fitness = -np.inf
        self.stagnation_count = 0
        self.recovery_steps_left = 0
        self.step_count = 0

    def act(self, observation) -> int:
        """Choose the next phase."""
        obs = np.asarray(observation, dtype=np.float32)
        base_scores = self._base_scores(obs)
        candidate_scores = self._search_scores(base_scores)
        cache_scores = self._cache_scores(obs)
        blended = (1.0 - self._cache_weight()) * candidate_scores + self._cache_weight() * cache_scores

        if self.action is not None:
            blended[self.action] -= self.switch_penalty
            if self.recovery_steps_left > 0:
                blended[self.action] -= self.switch_penalty

        action = int(np.argmax(blended))
        self.action = action
        self.last_scores = blended.astype(np.float32)
        self.last_state_key = self._state_key(obs)
        self.last_observation = obs.copy()
        self.last_queue = self._metric(obs, "queue")
        self.last_delay = self._metric(obs, "delay")
        self.step_count += 1
        return action

    def learn(self, next_observation, reward=None, info=None, done: bool = False) -> None:
        """Update search memory, cache, and recovery state."""
        if self.action is None or self.last_state_key is None:
            return

        obs = np.asarray(next_observation, dtype=np.float32)
        realized_fitness = self._fitness(obs)
        self._update_elite(self.last_scores, realized_fitness)
        self._update_cache(self.last_state_key, self.action, realized_fitness)
        self._update_recovery(obs, realized_fitness)

        if done:
            self.action = None
            self.last_state_key = None
            self.last_observation = None

    def _base_scores(self, obs: np.ndarray) -> np.ndarray:
        phase_offset = self.num_actions + 10
        phase_demand = obs[phase_offset : phase_offset + self.num_actions]
        phase_delay = obs[phase_offset + self.num_actions : phase_offset + 2 * self.num_actions]
        throughput = self._metric(obs, "throughput")
        pressure = self._metric(obs, "pressure")

        base = (
            self.weight_queue * phase_demand
            + self.weight_delay * phase_delay
            + self.weight_pressure * pressure
            + self.weight_throughput * throughput
        )
        return base.astype(np.float32)

    def _search_scores(self, base_scores: np.ndarray) -> np.ndarray:
        sigma = self.base_sigma * (2.0 if self.recovery_steps_left > 0 else 1.0)
        best = self._best_vector(base_scores)
        new_population = np.zeros_like(self.population)

        for idx in range(self.population_size):
            rand1, rand2 = self.population[self.rng.integers(self.population_size)], self.population[
                self.rng.integers(self.population_size)
            ]
            noise = self.rng.normal(0.0, sigma, size=self.num_actions)
            new_population[idx] = (
                base_scores
                + self.alpha * self.rng.random() * (best - self.population[idx])
                + self.beta * self.rng.random() * (rand1 - rand2)
                + noise
            )

        if self.recovery_steps_left > 0:
            recovery_count = max(1, self.population_size // 3)
            for idx in range(recovery_count):
                new_population[idx] = base_scores + self.rng.normal(0.0, sigma * 1.5, size=self.num_actions)

        self.population = new_population.astype(np.float32)
        return self.population.mean(axis=0)

    def _best_vector(self, base_scores: np.ndarray) -> np.ndarray:
        valid_scores = self.elite_scores[np.isfinite(self.elite_scores)]
        if valid_scores.size == 0:
            return base_scores
        elite = self.elite_population[np.argmax(self.elite_scores)]
        return elite

    def _cache_scores(self, obs: np.ndarray) -> np.ndarray:
        key = self._state_key(obs)
        entry = self.cache.get(key)
        if entry is None:
            return self._base_scores(obs)
        self.cache.move_to_end(key)
        return entry["values"].copy()

    def _cache_weight(self) -> float:
        if self.recovery_steps_left > 0:
            return self.base_cache_weight * 0.25
        return self.base_cache_weight

    def _update_elite(self, score_vector: np.ndarray, fitness: float) -> None:
        replace_idx = int(np.argmin(self.elite_scores))
        if fitness > self.elite_scores[replace_idx]:
            self.elite_scores[replace_idx] = fitness
            self.elite_population[replace_idx] = score_vector

    def _update_cache(self, key: tuple[int, ...], action: int, fitness: float) -> None:
        entry = self.cache.get(key)
        if entry is None:
            values = np.full(self.num_actions, fitness / max(1, self.num_actions), dtype=np.float32)
            values[action] = fitness
            self.cache[key] = {"values": values, "visits": 1}
        else:
            values = entry["values"]
            values[action] = 0.7 * float(values[action]) + 0.3 * fitness
            entry["visits"] = int(entry["visits"]) + 1
            self.cache.move_to_end(key)

        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def _update_recovery(self, obs: np.ndarray, fitness: float) -> None:
        queue = self._metric(obs, "queue")
        delay = self._metric(obs, "delay")

        improved = fitness > self.best_fitness + 1e-6
        if improved:
            self.best_fitness = fitness
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        queue_worsened = queue > max(self.recovery_queue_threshold, self.last_queue)
        delay_worsened = delay > max(self.recovery_delay_threshold, self.last_delay)
        if self.stagnation_count >= self.recovery_window or queue_worsened or delay_worsened:
            self.recovery_steps_left = self.recovery_window
        elif self.recovery_steps_left > 0:
            self.recovery_steps_left -= 1

    def _fitness(self, obs: np.ndarray) -> float:
        queue = self._metric(obs, "queue")
        delay = self._metric(obs, "delay")
        speed = self._metric(obs, "speed")
        throughput = self._metric(obs, "throughput")
        return float(
            self.fitness_weight_throughput * throughput
            + self.fitness_weight_speed * speed
            - self.fitness_weight_queue * queue
            - self.fitness_weight_delay * delay
        )

    def _state_key(self, obs: np.ndarray) -> tuple[int, ...]:
        return tuple(np.clip((obs * 4).astype(int), 0, 4).tolist())

    def _metric(self, obs: np.ndarray, name: str) -> float:
        offset = self.num_actions + 1
        mapping = {
            "queue": offset,
            "delay": offset + 1,
            "speed": offset + 2,
            "pressure": offset + 3,
            "throughput": offset + 6,
        }
        return float(obs[mapping[name]])
