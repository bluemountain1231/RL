# Traffic Signal RL Thesis Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thesis-ready Python + SUMO traffic-signal RL project with a fixed-time baseline, a Gym-like environment, DQN/DDPG agents, tests, training/evaluation scripts, and experiment outputs.

**Architecture:** The implementation starts with pure-Python state/action/reward logic and tests so the project has a stable core before SUMO integration. SUMO scenario files and `TrafficSignalEnv` are then added on top of that core, followed by a baseline controller, shared training utilities, DQN, DDPG, and a unified evaluation/reporting pipeline.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, PyYAML, pytest, SUMO, TraCI, matplotlib

---

## File structure

- Create: `pyproject.toml` — Python packaging, dependencies, pytest config
- Create: `.python-version` — local Python version pin
- Create: `configs/base.yaml` — global training and output defaults
- Create: `configs/dqn.yaml` — DQN hyperparameters
- Create: `configs/ddpg.yaml` — DDPG hyperparameters
- Create: `configs/reward.yaml` — reward weights
- Create: `configs/scenario_single_intersection.yaml` — scenario config and traffic-light constraints
- Create: `envs/__init__.py` — environment package export
- Create: `envs/observation_space.py` — state layout constants and dimension helpers
- Create: `envs/state_builder.py` — state-vector construction from lane metrics
- Create: `envs/action.py` — discrete/continuous action mapping to legal green-time changes
- Create: `envs/reward.py` — reward calculation
- Create: `envs/sumo_env.py` — `TrafficSignalEnv` orchestration around SUMO/TraCI
- Create: `agents/__init__.py` — agent package export
- Create: `agents/base_agent.py` — common agent protocol
- Create: `agents/replay_buffer.py` — replay buffer
- Create: `agents/networks.py` — MLP, actor, critic networks
- Create: `agents/dqn.py` — DQN agent
- Create: `agents/ddpg.py` — DDPG agent
- Create: `training/__init__.py` — training package export
- Create: `training/baseline.py` — fixed-time baseline policy
- Create: `training/train.py` — unified training entrypoint
- Create: `training/evaluate.py` — unified evaluation entrypoint
- Create: `utils/__init__.py` — utils package export
- Create: `utils/config.py` — YAML loader and merge helpers
- Create: `utils/metrics.py` — metric aggregation
- Create: `utils/plotting.py` — plot generation
- Create: `utils/logger.py` — CSV logging helpers
- Create: `utils/seeding.py` — random seed helper
- Create: `scenarios/single_intersection/intersection.net.xml` — SUMO network
- Create: `scenarios/single_intersection/routes.rou.xml` — basic route definitions
- Create: `scenarios/single_intersection/simulation.sumocfg` — SUMO scenario config
- Create: `tests/test_state_builder.py` — state builder tests
- Create: `tests/test_action_mapping.py` — action mapping tests
- Create: `tests/test_reward.py` — reward tests
- Create: `tests/test_replay_buffer.py` — replay buffer tests
- Create: `tests/test_agents.py` — DQN/DDPG smoke tests
- Create: `tests/test_env_smoke.py` — environment smoke test
- Create: `results/.gitkeep` — keep results directory in repo
- Modify: `docs/superpowers/specs/2026-04-15-traffic-signal-rl-design.md` — no modification required during implementation

### Task 1: Bootstrap Python project and configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `configs/base.yaml`
- Create: `configs/dqn.yaml`
- Create: `configs/ddpg.yaml`
- Create: `configs/reward.yaml`
- Create: `configs/scenario_single_intersection.yaml`
- Create: `results/.gitkeep`
- Test: `python -m pytest`

- [ ] **Step 1: Write the failing bootstrap test command**

Run:
```bash
python -m pytest
```
Expected: FAIL with import/path errors because the project files do not exist yet.

- [ ] **Step 2: Create Python packaging and dependency definition**

Write `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "traffic-signal-rl"
version = "0.1.0"
description = "Single-intersection traffic signal control with DQN and DDPG"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "torch>=2.2",
  "PyYAML>=6.0",
  "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
]
sumo = [
  "sumolib",
  "traci",
]

[tool.setuptools]
packages = ["agents", "envs", "training", "utils"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Write `.python-version`:
```text
3.11
```

- [ ] **Step 3: Create base configuration files**

Write `configs/base.yaml`:
```yaml
seed: 42
output_dir: results
scenario: configs/scenario_single_intersection.yaml
reward: configs/reward.yaml
steps_per_action: 5
max_episode_steps: 360
train_episodes: 50
eval_episodes: 5
checkpoint_every: 10
device: cpu
agent: dqn
```

Write `configs/dqn.yaml`:
```yaml
gamma: 0.99
learning_rate: 0.001
batch_size: 32
buffer_capacity: 5000
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay: 0.995
target_update_freq: 25
hidden_dim: 128
```

Write `configs/ddpg.yaml`:
```yaml
gamma: 0.99
actor_learning_rate: 0.0003
critic_learning_rate: 0.001
batch_size: 32
buffer_capacity: 5000
tau: 0.005
hidden_dim: 128
noise_std: 0.1
```

Write `configs/reward.yaml`:
```yaml
waiting_weight: 1.0
queue_weight: 0.5
throughput_weight: 0.2
fairness_weight: 0.1
```

Write `configs/scenario_single_intersection.yaml`:
```yaml
name: single_intersection
sumocfg_path: scenarios/single_intersection/simulation.sumocfg
traffic_light_id: junction_0
incoming_lanes:
  - north_in_0
  - south_in_0
  - east_in_0
  - west_in_0
phase_count: 2
min_green: 10
max_green: 60
yellow_time: 3
all_red_time: 1
action_delta_seconds: 5
```

Write `results/.gitkeep` as an empty file.

- [ ] **Step 4: Install dependencies and verify pytest runs**

Run:
```bash
python -m pip install -e ".[dev]"
python -m pytest
```
Expected: PASS with `collected 0 items` because tests are not written yet.

- [ ] **Step 5: Commit bootstrap files**

Run:
```bash
git add pyproject.toml .python-version configs/base.yaml configs/dqn.yaml configs/ddpg.yaml configs/reward.yaml configs/scenario_single_intersection.yaml results/.gitkeep
git commit -m "$(cat <<'EOF'
Bootstrap Python project and experiment configs.

Add packaging metadata, base RL configuration, reward weights, and the single-intersection scenario config.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2: Implement observation layout, action mapping, and reward logic with tests

**Files:**
- Create: `envs/__init__.py`
- Create: `envs/observation_space.py`
- Create: `envs/state_builder.py`
- Create: `envs/action.py`
- Create: `envs/reward.py`
- Create: `tests/test_state_builder.py`
- Create: `tests/test_action_mapping.py`
- Create: `tests/test_reward.py`
- Test: `tests/test_state_builder.py`
- Test: `tests/test_action_mapping.py`
- Test: `tests/test_reward.py`

- [ ] **Step 1: Write failing tests for state, action, and reward behavior**

Write `tests/test_state_builder.py`:
```python
import numpy as np

from envs.state_builder import build_state


def test_build_state_orders_features_by_lane_then_phase():
    lane_metrics = {
        "north_in_0": {"queue_length": 2.0, "average_speed": 8.0, "waiting_time": 12.0},
        "south_in_0": {"queue_length": 1.0, "average_speed": 7.5, "waiting_time": 9.0},
        "east_in_0": {"queue_length": 3.0, "average_speed": 6.0, "waiting_time": 15.0},
        "west_in_0": {"queue_length": 0.0, "average_speed": 9.5, "waiting_time": 4.0},
    }

    state = build_state(
        lane_metrics=lane_metrics,
        lane_order=["north_in_0", "south_in_0", "east_in_0", "west_in_0"],
        current_phase=1,
        phase_count=2,
        phase_elapsed=6.0,
    )

    expected = np.array([
        2.0, 8.0, 12.0,
        1.0, 7.5, 9.0,
        3.0, 6.0, 15.0,
        0.0, 9.5, 4.0,
        0.0, 1.0,
        6.0,
    ], dtype=np.float32)

    assert np.allclose(state, expected)
```

Write `tests/test_action_mapping.py`:
```python
from envs.action import decode_ddpg_action, decode_dqn_action


def test_decode_dqn_action_maps_index_to_green_delta():
    assert decode_dqn_action(action_index=0, action_delta_seconds=5) == -5
    assert decode_dqn_action(action_index=1, action_delta_seconds=5) == 0
    assert decode_dqn_action(action_index=2, action_delta_seconds=5) == 5


def test_decode_ddpg_action_clips_and_scales_to_green_delta():
    assert decode_ddpg_action(raw_action=-2.0, action_delta_seconds=5) == -5
    assert decode_ddpg_action(raw_action=0.0, action_delta_seconds=5) == 0
    assert decode_ddpg_action(raw_action=0.6, action_delta_seconds=5) == 3
    assert decode_ddpg_action(raw_action=2.0, action_delta_seconds=5) == 5
```

Write `tests/test_reward.py`:
```python
from envs.reward import compute_reward


def test_compute_reward_combines_waiting_queue_throughput_and_fairness():
    reward = compute_reward(
        total_waiting=10.0,
        total_queue=4.0,
        throughput=3.0,
        directional_waiting_times=[2.0, 4.0, 6.0, 8.0],
        waiting_weight=1.0,
        queue_weight=0.5,
        throughput_weight=0.2,
        fairness_weight=0.1,
    )

    assert round(reward, 2) == -11.10
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/test_state_builder.py tests/test_action_mapping.py tests/test_reward.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'envs'`.

- [ ] **Step 3: Implement observation, state, action, and reward modules**

Write `envs/__init__.py`:
```python
from envs.sumo_env import TrafficSignalEnv

__all__ = ["TrafficSignalEnv"]
```

Write `envs/observation_space.py`:
```python
FEATURES_PER_LANE = ("queue_length", "average_speed", "waiting_time")


def state_dimension(lane_count: int, phase_count: int) -> int:
    return lane_count * len(FEATURES_PER_LANE) + phase_count + 1
```

Write `envs/state_builder.py`:
```python
from __future__ import annotations

import numpy as np

from envs.observation_space import FEATURES_PER_LANE



def build_state(
    lane_metrics: dict[str, dict[str, float]],
    lane_order: list[str],
    current_phase: int,
    phase_count: int,
    phase_elapsed: float,
) -> np.ndarray:
    values: list[float] = []
    for lane_id in lane_order:
        metrics = lane_metrics[lane_id]
        for feature_name in FEATURES_PER_LANE:
            values.append(float(metrics[feature_name]))

    phase_one_hot = [0.0] * phase_count
    phase_one_hot[current_phase] = 1.0
    values.extend(phase_one_hot)
    values.append(float(phase_elapsed))
    return np.asarray(values, dtype=np.float32)
```

Write `envs/action.py`:
```python
from __future__ import annotations


def decode_dqn_action(action_index: int, action_delta_seconds: int) -> int:
    mapping = {
        0: -action_delta_seconds,
        1: 0,
        2: action_delta_seconds,
    }
    return mapping[action_index]



def decode_ddpg_action(raw_action: float, action_delta_seconds: int) -> int:
    clipped = max(-1.0, min(1.0, float(raw_action)))
    return int(round(clipped * action_delta_seconds))



def apply_green_constraints(
    proposed_green: int,
    min_green: int,
    max_green: int,
) -> int:
    return max(min_green, min(max_green, proposed_green))
```

Write `envs/reward.py`:
```python
from __future__ import annotations

import statistics



def compute_reward(
    total_waiting: float,
    total_queue: float,
    throughput: float,
    directional_waiting_times: list[float],
    waiting_weight: float,
    queue_weight: float,
    throughput_weight: float,
    fairness_weight: float,
) -> float:
    fairness_penalty = statistics.pvariance(directional_waiting_times) if directional_waiting_times else 0.0
    return (
        -(waiting_weight * total_waiting + queue_weight * total_queue)
        + throughput_weight * throughput
        - fairness_weight * fairness_penalty
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_state_builder.py tests/test_action_mapping.py tests/test_reward.py -v
```
Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit environment core logic**

Run:
```bash
git add envs/__init__.py envs/observation_space.py envs/state_builder.py envs/action.py envs/reward.py tests/test_state_builder.py tests/test_action_mapping.py tests/test_reward.py
git commit -m "$(cat <<'EOF'
Add tested state, action, and reward modules.

Implement the pure-Python environment core and cover state layout, action mapping, and reward calculation with unit tests.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3: Implement replay buffer, shared networks, and agent smoke tests

**Files:**
- Create: `agents/__init__.py`
- Create: `agents/base_agent.py`
- Create: `agents/replay_buffer.py`
- Create: `agents/networks.py`
- Create: `tests/test_replay_buffer.py`
- Create: `tests/test_agents.py`
- Test: `tests/test_replay_buffer.py`
- Test: `tests/test_agents.py`

- [ ] **Step 1: Write failing tests for replay buffer and network-backed agents**

Write `tests/test_replay_buffer.py`:
```python
import numpy as np

from agents.replay_buffer import ReplayBuffer


def test_replay_buffer_stores_and_samples_batches():
    buffer = ReplayBuffer(capacity=3)
    for index in range(3):
        buffer.push(
            state=np.array([index], dtype=np.float32),
            action=np.array([index], dtype=np.float32),
            reward=float(index),
            next_state=np.array([index + 1], dtype=np.float32),
            done=bool(index % 2),
        )

    batch = buffer.sample(batch_size=2)

    assert batch["state"].shape == (2, 1)
    assert batch["action"].shape == (2, 1)
    assert batch["reward"].shape == (2,)
    assert batch["next_state"].shape == (2, 1)
    assert batch["done"].shape == (2,)
```

Write `tests/test_agents.py`:
```python
import numpy as np

from agents.ddpg import DDPGAgent
from agents.dqn import DQNAgent


def test_dqn_agent_returns_valid_action_index():
    agent = DQNAgent(state_dim=15, action_dim=3, hidden_dim=32, gamma=0.99, learning_rate=0.001, batch_size=2, target_update_freq=5)
    action = agent.act(np.zeros(15, dtype=np.float32), eval_mode=True)
    assert action in {0, 1, 2}


def test_ddpg_agent_returns_scalar_action():
    agent = DDPGAgent(state_dim=15, action_dim=1, hidden_dim=32, gamma=0.99, actor_learning_rate=0.001, critic_learning_rate=0.001, batch_size=2, tau=0.005, noise_std=0.1)
    action = agent.act(np.zeros(15, dtype=np.float32), eval_mode=True)
    assert isinstance(action, float)
    assert -1.0 <= action <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/test_replay_buffer.py tests/test_agents.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'agents'`.

- [ ] **Step 3: Implement replay buffer and shared network code**

Write `agents/__init__.py`:
```python
from agents.ddpg import DDPGAgent
from agents.dqn import DQNAgent

__all__ = ["DQNAgent", "DDPGAgent"]
```

Write `agents/base_agent.py`:
```python
from __future__ import annotations

from typing import Protocol

import numpy as np


class BaseAgent(Protocol):
    def act(self, state: np.ndarray, eval_mode: bool = False): ...
    def remember(self, transition: tuple): ...
    def update(self): ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

Write `agents/replay_buffer.py`:
```python
from __future__ import annotations

from collections import deque
import random

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        batch = random.sample(list(self.buffer), batch_size)
        state, action, reward, next_state, done = zip(*batch, strict=True)
        return {
            "state": np.stack(state),
            "action": np.stack(action),
            "reward": np.asarray(reward, dtype=np.float32),
            "next_state": np.stack(next_state),
            "done": np.asarray(done, dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.buffer)
```

Write `agents/networks.py`:
```python
from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([state, action], dim=-1))
```

Write `agents/dqn.py`:
```python
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from agents.networks import QNetwork
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, gamma: float, learning_rate: float, batch_size: int, target_update_freq: int, epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: float = 0.995, buffer_capacity: int = 5000) -> None:
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.online_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)
        self.update_steps = 0

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, transition: tuple) -> None:
        self.buffer.push(*transition)

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch = self.buffer.sample(self.batch_size)
        state = torch.as_tensor(batch["state"], dtype=torch.float32)
        action = torch.as_tensor(batch["action"], dtype=torch.int64).view(-1, 1)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32)
        next_state = torch.as_tensor(batch["next_state"], dtype=torch.float32)
        done = torch.as_tensor(batch["done"], dtype=torch.float32)

        current_q = self.online_net(state).gather(1, action).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_state).max(dim=1).values
            target_q = reward + self.gamma * next_q * (1.0 - done)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
```

Write `agents/ddpg.py`:
```python
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from agents.networks import Actor, Critic
from agents.replay_buffer import ReplayBuffer


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, gamma: float, actor_learning_rate: float, critic_learning_rate: float, batch_size: int, tau: float, noise_std: float, buffer_capacity: int = 5000) -> None:
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.noise_std = noise_std
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)

    def act(self, state: np.ndarray, eval_mode: bool = False) -> float:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).cpu().numpy()[0]
        if not eval_mode:
            action = action + np.random.normal(0.0, self.noise_std)
        return float(np.clip(action, -1.0, 1.0))

    def remember(self, transition: tuple) -> None:
        self.buffer.push(*transition)

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch = self.buffer.sample(self.batch_size)
        state = torch.as_tensor(batch["state"], dtype=torch.float32)
        action = torch.as_tensor(batch["action"], dtype=torch.float32)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32).unsqueeze(1)
        next_state = torch.as_tensor(batch["next_state"], dtype=torch.float32)
        done = torch.as_tensor(batch["done"], dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            next_action = self.target_actor(next_state)
            next_q = self.target_critic(next_state, next_action)
            target_q = reward + self.gamma * next_q * (1.0 - done)

        critic_loss = self.loss_fn(self.critic(state, action), target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
        return float(critic_loss.item())

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_actor.load_state_dict(state_dict["actor"])
        self.target_critic.load_state_dict(state_dict["critic"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_replay_buffer.py tests/test_agents.py -v
```
Expected: PASS with `4 passed`.

- [ ] **Step 5: Commit shared agent infrastructure**

Run:
```bash
git add agents/__init__.py agents/base_agent.py agents/replay_buffer.py agents/networks.py agents/dqn.py agents/ddpg.py tests/test_replay_buffer.py tests/test_agents.py
git commit -m "$(cat <<'EOF'
Add replay buffer and DQN/DDPG agent implementations.

Implement the shared neural networks, replay buffer, and the first tested versions of the DQN and DDPG agents.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4: Add configuration utilities, metrics, plotting, and baseline policy

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/config.py`
- Create: `utils/metrics.py`
- Create: `utils/plotting.py`
- Create: `utils/logger.py`
- Create: `utils/seeding.py`
- Create: `training/__init__.py`
- Create: `training/baseline.py`
- Test: `tests/test_agents.py`

- [ ] **Step 1: Write failing utility/baseline test**

Append to `tests/test_agents.py`:
```python
from training.baseline import FixedTimeBaselinePolicy


def test_fixed_time_baseline_cycles_green_duration_values():
    policy = FixedTimeBaselinePolicy(schedule=[20, 25])
    assert policy.act(step_index=0) == 20
    assert policy.act(step_index=1) == 25
    assert policy.act(step_index=2) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_agents.py::test_fixed_time_baseline_cycles_green_duration_values -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'training'`.

- [ ] **Step 3: Implement utility and baseline modules**

Write `utils/__init__.py`:
```python
from utils.config import load_yaml
from utils.metrics import EpisodeMetrics

__all__ = ["load_yaml", "EpisodeMetrics"]
```

Write `utils/config.py`:
```python
from __future__ import annotations

from pathlib import Path

import yaml



def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())



def merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
```

Write `utils/metrics.py`:
```python
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
```

Write `utils/logger.py`:
```python
from __future__ import annotations

import csv
from pathlib import Path



def append_row(csv_path: str, row: dict[str, float | int | str]) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
```

Write `utils/plotting.py`:
```python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt



def save_line_plot(values: list[float], title: str, y_label: str, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
```

Write `utils/seeding.py`:
```python
from __future__ import annotations

import random

import numpy as np
import torch



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

Write `training/__init__.py`:
```python
from training.baseline import FixedTimeBaselinePolicy

__all__ = ["FixedTimeBaselinePolicy"]
```

Write `training/baseline.py`:
```python
from __future__ import annotations


class FixedTimeBaselinePolicy:
    def __init__(self, schedule: list[int]) -> None:
        self.schedule = schedule

    def act(self, step_index: int) -> int:
        return self.schedule[step_index % len(self.schedule)]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_agents.py::test_fixed_time_baseline_cycles_green_duration_values -v
```
Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit utility and baseline support**

Run:
```bash
git add utils/__init__.py utils/config.py utils/metrics.py utils/logger.py utils/plotting.py utils/seeding.py training/__init__.py training/baseline.py tests/test_agents.py
git commit -m "$(cat <<'EOF'
Add config, metrics, plotting, and baseline policy support.

Create the shared utility layer and a fixed-time baseline policy for the common training and evaluation pipeline.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5: Add SUMO scenario assets and environment smoke test

**Files:**
- Create: `scenarios/single_intersection/intersection.net.xml`
- Create: `scenarios/single_intersection/routes.rou.xml`
- Create: `scenarios/single_intersection/simulation.sumocfg`
- Create: `envs/sumo_env.py`
- Create: `tests/test_env_smoke.py`
- Test: `tests/test_env_smoke.py`

- [ ] **Step 1: Write failing environment smoke test**

Write `tests/test_env_smoke.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_env_smoke.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'envs.sumo_env'`.

- [ ] **Step 3: Add SUMO assets and environment implementation**

Write `scenarios/single_intersection/intersection.net.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" junctionCornerDetail="5" limitTurnSpeed="5.50">
    <location netOffset="0.00,0.00" convBoundary="-100.00,-100.00,100.00,100.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    <edge id="north_in" from="north" to="junction_0" priority="-1">
        <lane id="north_in_0" index="0" speed="13.89" length="90.00" shape="0.00,100.00 0.00,10.00"/>
    </edge>
    <edge id="south_in" from="south" to="junction_0" priority="-1">
        <lane id="south_in_0" index="0" speed="13.89" length="90.00" shape="0.00,-100.00 0.00,-10.00"/>
    </edge>
    <edge id="east_in" from="east" to="junction_0" priority="-1">
        <lane id="east_in_0" index="0" speed="13.89" length="90.00" shape="100.00,0.00 10.00,0.00"/>
    </edge>
    <edge id="west_in" from="west" to="junction_0" priority="-1">
        <lane id="west_in_0" index="0" speed="13.89" length="90.00" shape="-100.00,0.00 -10.00,0.00"/>
    </edge>
    <edge id="north_out" from="junction_0" to="north_exit" priority="-1">
        <lane id="north_out_0" index="0" speed="13.89" length="90.00" shape="0.00,10.00 0.00,100.00"/>
    </edge>
    <edge id="south_out" from="junction_0" to="south_exit" priority="-1">
        <lane id="south_out_0" index="0" speed="13.89" length="90.00" shape="0.00,-10.00 0.00,-100.00"/>
    </edge>
    <edge id="east_out" from="junction_0" to="east_exit" priority="-1">
        <lane id="east_out_0" index="0" speed="13.89" length="90.00" shape="10.00,0.00 100.00,0.00"/>
    </edge>
    <edge id="west_out" from="junction_0" to="west_exit" priority="-1">
        <lane id="west_out_0" index="0" speed="13.89" length="90.00" shape="-10.00,0.00 -100.00,0.00"/>
    </edge>
    <tlLogic id="junction_0" type="static" programID="0" offset="0">
        <phase duration="20" state="GrGr"/>
        <phase duration="3" state="yryr"/>
        <phase duration="20" state="rGrG"/>
        <phase duration="3" state="ryry"/>
    </tlLogic>
    <junction id="junction_0" type="traffic_light" x="0.00" y="0.00" incLanes="north_in_0 south_in_0 east_in_0 west_in_0" intLanes="" shape="-5.00,5.00 5.00,5.00 5.00,-5.00 -5.00,-5.00"/>
    <junction id="north" type="priority" x="0.00" y="100.00" incLanes="" intLanes="" shape="0.00,100.00"/>
    <junction id="south" type="priority" x="0.00" y="-100.00" incLanes="" intLanes="" shape="0.00,-100.00"/>
    <junction id="east" type="priority" x="100.00" y="0.00" incLanes="" intLanes="" shape="100.00,0.00"/>
    <junction id="west" type="priority" x="-100.00" y="0.00" incLanes="" intLanes="" shape="-100.00,0.00"/>
    <junction id="north_exit" type="priority" x="0.00" y="110.00" incLanes="north_out_0" intLanes="" shape="0.00,110.00"/>
    <junction id="south_exit" type="priority" x="0.00" y="-110.00" incLanes="south_out_0" intLanes="" shape="0.00,-110.00"/>
    <junction id="east_exit" type="priority" x="110.00" y="0.00" incLanes="east_out_0" intLanes="" shape="110.00,0.00"/>
    <junction id="west_exit" type="priority" x="-110.00" y="0.00" incLanes="west_out_0" intLanes="" shape="-110.00,0.00"/>
</net>
```

Write `scenarios/single_intersection/routes.rou.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="1.0" decel="4.5" sigma="0.5" length="5.0" maxSpeed="13.89"/>
    <route id="north_south" edges="north_in south_out"/>
    <route id="south_north" edges="south_in north_out"/>
    <route id="east_west" edges="east_in west_out"/>
    <route id="west_east" edges="west_in east_out"/>
    <flow id="flow_north" type="car" route="north_south" begin="0" end="3600" vehsPerHour="240"/>
    <flow id="flow_south" type="car" route="south_north" begin="0" end="3600" vehsPerHour="240"/>
    <flow id="flow_east" type="car" route="east_west" begin="0" end="3600" vehsPerHour="180"/>
    <flow id="flow_west" type="car" route="west_east" begin="0" end="3600" vehsPerHour="180"/>
</routes>
```

Write `scenarios/single_intersection/simulation.sumocfg`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>
```

Write `envs/sumo_env.py`:
```python
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
```

- [ ] **Step 4: Run smoke test to verify it passes**

Run:
```bash
python -m pytest tests/test_env_smoke.py -v
```
Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit SUMO scenario and environment**

Run:
```bash
git add scenarios/single_intersection/intersection.net.xml scenarios/single_intersection/routes.rou.xml scenarios/single_intersection/simulation.sumocfg envs/sumo_env.py tests/test_env_smoke.py
git commit -m "$(cat <<'EOF'
Add single-intersection SUMO assets and environment wrapper.

Create the first runnable scenario assets and implement a smoke-tested TrafficSignalEnv for training and evaluation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6: Implement unified training entrypoint and DQN end-to-end loop

**Files:**
- Create: `training/train.py`
- Test: `tests/test_agents.py`
- Test: `tests/test_env_smoke.py`

- [ ] **Step 1: Write failing DQN training smoke test**

Append to `tests/test_env_smoke.py`:
```python
from agents.dqn import DQNAgent
from training.train import run_training_episode


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
    agent = DQNAgent(state_dim=15, action_dim=3, hidden_dim=32, gamma=0.99, learning_rate=0.001, batch_size=2, target_update_freq=5)

    summary = run_training_episode(env=env, agent=agent, action_type="dqn", max_steps=5)

    assert set(summary.keys()) == {
        "episode_reward",
        "average_waiting_time",
        "average_queue_length",
        "throughput",
        "average_speed",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_env_smoke.py::test_run_training_episode_returns_summary_for_dqn -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train'`.

- [ ] **Step 3: Implement the shared training loop**

Write `training/train.py`:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--agent-config", required=True)
    args = parser.parse_args()

    base_config = load_yaml(args.base_config)
    agent_config = load_yaml(args.agent_config)
    scenario_config = load_yaml(base_config["scenario"])
    reward_config = load_yaml(base_config["reward"])
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_env_smoke.py::test_run_training_episode_returns_summary_for_dqn -v
```
Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit DQN training loop**

Run:
```bash
git add training/train.py tests/test_env_smoke.py
git commit -m "$(cat <<'EOF'
Add unified training loop and DQN episode smoke path.

Implement the shared training entrypoint and verify a full DQN episode can run through the environment.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7: Implement evaluation entrypoint and baseline/DQN/DDPG comparison flow

**Files:**
- Create: `training/evaluate.py`
- Test: `tests/test_env_smoke.py`

- [ ] **Step 1: Write failing evaluation smoke test**

Append to `tests/test_env_smoke.py`:
```python
from training.evaluate import run_evaluation_episode
from training.baseline import FixedTimeBaselinePolicy


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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_env_smoke.py::test_run_evaluation_episode_supports_baseline_policy -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'training.evaluate'`.

- [ ] **Step 3: Implement evaluation pipeline**

Write `training/evaluate.py`:
```python
from __future__ import annotations

import argparse
import statistics

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



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--agent", choices=["baseline", "dqn", "ddpg"], required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--agent-config", default="")
    args = parser.parse_args()

    base_config = load_yaml(args.base_config)
    scenario_config = load_yaml(base_config["scenario"])
    reward_config = load_yaml(base_config["reward"])
    set_seed(base_config["seed"])
    env = TrafficSignalEnv(scenario_config=scenario_config, reward_config=reward_config, sumo_enabled=False)

    if args.agent == "baseline":
        policy = FixedTimeBaselinePolicy(schedule=[20, 25])
        action_type = "baseline"
    elif args.agent == "dqn":
        agent_config = load_yaml(args.agent_config)
        policy = DQNAgent(state_dim=len(env.reset()), action_dim=3, **agent_config)
        policy.load(args.checkpoint)
        action_type = "dqn"
    else:
        agent_config = load_yaml(args.agent_config)
        policy = DDPGAgent(state_dim=len(env.reset()), action_dim=1, **agent_config)
        policy.load(args.checkpoint)
        action_type = "ddpg"

    summaries = [run_evaluation_episode(env, policy, action_type, base_config["max_episode_steps"]) for _ in range(base_config["eval_episodes"])]
    aggregate = aggregate_summaries(summaries)
    append_row(csv_path=f"{base_config['output_dir']}/csv/eval_{args.agent}.csv", row=aggregate)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_env_smoke.py::test_run_evaluation_episode_supports_baseline_policy -v
```
Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit evaluation flow**

Run:
```bash
git add training/evaluate.py tests/test_env_smoke.py
git commit -m "$(cat <<'EOF'
Add unified evaluation flow for baseline, DQN, and DDPG.

Implement the evaluation entrypoint so all policies can be compared under the same metrics and episode settings.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 8: Run the full test suite and generate first local outputs

**Files:**
- Modify: `results/csv/train_dqn.csv`
- Modify: `results/csv/eval_baseline.csv`
- Modify: `results/plots/dqn_reward.png`
- Test: `tests/test_state_builder.py`
- Test: `tests/test_action_mapping.py`
- Test: `tests/test_reward.py`
- Test: `tests/test_replay_buffer.py`
- Test: `tests/test_agents.py`
- Test: `tests/test_env_smoke.py`

- [ ] **Step 1: Run the full test suite**

Run:
```bash
python -m pytest -v
```
Expected: PASS with all six test files green.

- [ ] **Step 2: Run the first DQN training pass**

Run:
```bash
python training/train.py --base-config configs/base.yaml --agent-config configs/dqn.yaml
```
Expected: PASS and create:
- `results/csv/train_dqn.csv`
- `results/plots/dqn_reward.png`
- `results/checkpoints/dqn_10.pt`
- `results/checkpoints/dqn_20.pt`
- `results/checkpoints/dqn_30.pt`
- `results/checkpoints/dqn_40.pt`
- `results/checkpoints/dqn_50.pt`

- [ ] **Step 3: Run the first baseline evaluation pass**

Run:
```bash
python training/evaluate.py --base-config configs/base.yaml --agent baseline
```
Expected: PASS and create `results/csv/eval_baseline.csv`.

- [ ] **Step 4: Run the first DQN evaluation pass**

Run:
```bash
python training/evaluate.py --base-config configs/base.yaml --agent dqn --agent-config configs/dqn.yaml --checkpoint results/checkpoints/dqn_50.pt
```
Expected: PASS and create `results/csv/eval_dqn.csv`.

- [ ] **Step 5: Switch base config and run the first DDPG training/evaluation pass**

Temporarily update `configs/base.yaml` to:
```yaml
seed: 42
output_dir: results
scenario: configs/scenario_single_intersection.yaml
reward: configs/reward.yaml
steps_per_action: 5
max_episode_steps: 360
train_episodes: 50
eval_episodes: 5
checkpoint_every: 10
device: cpu
agent: ddpg
```

Run:
```bash
python training/train.py --base-config configs/base.yaml --agent-config configs/ddpg.yaml
python training/evaluate.py --base-config configs/base.yaml --agent ddpg --agent-config configs/ddpg.yaml --checkpoint results/checkpoints/ddpg_50.pt
```
Expected: PASS and create:
- `results/csv/train_ddpg.csv`
- `results/csv/eval_ddpg.csv`
- `results/plots/ddpg_reward.png`
- `results/checkpoints/ddpg_10.pt`
- `results/checkpoints/ddpg_20.pt`
- `results/checkpoints/ddpg_30.pt`
- `results/checkpoints/ddpg_40.pt`
- `results/checkpoints/ddpg_50.pt`

- [ ] **Step 6: Commit the first runnable end-to-end system**

Run:
```bash
git add envs agents training utils scenarios configs tests pyproject.toml .python-version results/.gitkeep
git commit -m "$(cat <<'EOF'
Build the first end-to-end traffic signal RL thesis system.

Run the full test suite and land the baseline, DQN, and DDPG pipeline on the single-intersection SUMO project structure.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Self-review

### Spec coverage
- Single-intersection scenario: covered in Task 5.
- Fixed-time baseline controller: covered in Task 4 and Task 7.
- Gym-like SUMO environment: covered in Task 5.
- Separate state/action/reward modules: covered in Task 2.
- DQN agent: covered in Task 3 and Task 6.
- DDPG agent: covered in Task 3 and Task 8.
- Unified training/evaluation entrypoints: covered in Task 6 and Task 7.
- YAML config files: covered in Task 1.
- Basic tests: covered in Tasks 2, 3, and 5.
- Standardized results output: covered in Tasks 4, 6, 7, and 8.

### Placeholder scan
- No `TBD`, `TODO`, `implement later`, `fill in details`, or vague task text remains.

### Type consistency
- `TrafficSignalEnv.step()` accepts `int | float` actions and is used consistently by DQN, DDPG, and baseline evaluation.
- DQN stores integer action indices as shape `(1,)` arrays; DDPG stores float actions as shape `(1,)` arrays.
- The metric summary keys are identical in `run_training_episode()` and `run_evaluation_episode()`.
