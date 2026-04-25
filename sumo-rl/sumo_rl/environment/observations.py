"""Observation functions for traffic signals."""

from abc import abstractmethod
from typing import List, Optional

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )


class NOAObservationFunction(ObservationFunction):
    """Observation function tailored for Nutcracker-style online phase control."""

    def __init__(self, ts: TrafficSignal):
        """Initialize NOA observation state."""
        super().__init__(ts)
        self.phase_lane_map: Optional[List[List[str]]] = None
        self.prev_total_queue_raw: Optional[float] = None
        self.prev_avg_delay_raw: Optional[float] = None
        self.last_sim_step: Optional[float] = None
        self.last_observation: Optional[np.ndarray] = None

    def __call__(self) -> np.ndarray:
        """Return the NOA observation."""
        if self.last_sim_step == self.ts.env.sim_step and self.last_observation is not None:
            return self.last_observation.copy()

        self._ensure_phase_lane_map()

        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]

        lane_density = self.ts.get_lanes_density()
        lane_queue = self.ts.get_lanes_queue()
        lane_wait = self.ts.get_accumulated_waiting_time_per_lane()

        total_queue_raw = float(self.ts.get_total_queued())
        avg_delay_raw = float(np.mean(lane_wait)) if lane_wait else 0.0
        avg_speed = float(self.ts.get_average_speed())
        pressure_raw = float(self.ts.get_pressure())
        mean_density = float(np.mean(lane_density)) if lane_density else 0.0
        mean_queue = float(np.mean(lane_queue)) if lane_queue else 0.0

        queue_scale = max(1.0, float(len(self.ts.lanes)))
        throughput_proxy = 0.0
        delta_queue_raw = 0.0
        delta_delay_raw = 0.0
        if self.prev_total_queue_raw is not None:
            throughput_proxy = max(0.0, self.prev_total_queue_raw - total_queue_raw)
            delta_queue_raw = total_queue_raw - self.prev_total_queue_raw
        if self.prev_avg_delay_raw is not None:
            delta_delay_raw = avg_delay_raw - self.prev_avg_delay_raw

        phase_demand = []
        phase_delay = []
        phase_density = []
        for lanes in self.phase_lane_map:
            if not lanes:
                phase_demand.append(0.0)
                phase_delay.append(0.0)
                phase_density.append(0.0)
                continue

            indices = [self.ts.lanes.index(lane) for lane in lanes if lane in self.ts.lanes]
            if not indices:
                phase_demand.append(0.0)
                phase_delay.append(0.0)
                phase_density.append(0.0)
                continue

            demand = float(np.mean([(lane_queue[i] + lane_density[i]) / 2.0 for i in indices]))
            delay = float(np.mean([lane_wait[i] for i in indices]))
            density = float(np.mean([lane_density[i] for i in indices]))
            phase_demand.append(np.clip(demand, 0.0, 1.0))
            phase_delay.append(np.tanh(delay / 100.0))
            phase_density.append(np.clip(density, 0.0, 1.0))

        observation = np.array(
            phase_id
            + min_green
            + [
                np.tanh(total_queue_raw / queue_scale),
                np.tanh(avg_delay_raw / 100.0),
                np.clip(avg_speed, 0.0, 1.0),
                (np.tanh(pressure_raw / queue_scale) + 1.0) / 2.0,
                np.clip(mean_density, 0.0, 1.0),
                np.clip(mean_queue, 0.0, 1.0),
                np.tanh(throughput_proxy / queue_scale),
                (np.tanh(delta_queue_raw / queue_scale) + 1.0) / 2.0,
                (np.tanh(delta_delay_raw / 100.0) + 1.0) / 2.0,
            ]
            + phase_demand
            + phase_delay
            + phase_density,
            dtype=np.float32,
        )

        self.prev_total_queue_raw = total_queue_raw
        self.prev_avg_delay_raw = avg_delay_raw
        self.last_sim_step = self.ts.env.sim_step
        self.last_observation = observation
        return observation.copy()

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        size = 4 * self.ts.num_green_phases + 10
        return spaces.Box(low=np.zeros(size, dtype=np.float32), high=np.ones(size, dtype=np.float32), dtype=np.float32)

    def _ensure_phase_lane_map(self) -> None:
        """Build the mapping from green phases to incoming lanes on demand."""
        if self.phase_lane_map is not None:
            return

        controlled_links = self.ts.sumo.trafficlight.getControlledLinks(self.ts.id)
        green_phases = getattr(self.ts, "green_phases", [])
        if not green_phases:
            self.phase_lane_map = [list(self.ts.lanes) for _ in range(self.ts.num_green_phases)]
            return

        phase_lane_map = []
        for phase in green_phases:
            lanes = []
            for signal_idx, state in enumerate(phase.state):
                if signal_idx >= len(controlled_links) or state not in ("G", "g"):
                    continue
                for connection in controlled_links[signal_idx]:
                    incoming_lane = connection[0]
                    if incoming_lane in self.ts.lanes and incoming_lane not in lanes:
                        lanes.append(incoming_lane)
            phase_lane_map.append(lanes if lanes else list(self.ts.lanes))

        self.phase_lane_map = phase_lane_map
