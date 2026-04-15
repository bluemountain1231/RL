from __future__ import annotations

from collections import deque
import random
from typing import Any

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[tuple[np.ndarray, Any, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: Any, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        batch = random.sample(list(self.buffer), batch_size)
        state, action, reward, next_state, done = zip(*batch, strict=True)
        return {
            "state": np.stack(state),
            "action": np.asarray(action),
            "reward": np.asarray(reward, dtype=np.float32),
            "next_state": np.stack(next_state),
            "done": np.asarray(done, dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.buffer)
