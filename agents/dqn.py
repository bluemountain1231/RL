from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from agents.networks import QNetwork
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        gamma: float,
        learning_rate: float,
        batch_size: int,
        target_update_freq: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 5000,
    ) -> None:
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
