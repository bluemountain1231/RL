from __future__ import annotations

import numpy as np
import torch
from torch import nn

from agents.networks import Actor, Critic
from agents.replay_buffer import ReplayBuffer


class DDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        gamma: float,
        actor_learning_rate: float,
        critic_learning_rate: float,
        batch_size: int,
        tau: float,
        noise_std: float,
        buffer_capacity: int = 5000,
    ) -> None:
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

    def act(self, state: np.ndarray, eval_mode: bool = False) -> float | np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).cpu().numpy()
        if not eval_mode:
            action = action + np.random.normal(0.0, self.noise_std, size=action.shape)
        clipped = np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)
        if clipped.shape == (1,):
            return float(clipped[0])
        return clipped

    def remember(self, transition: tuple) -> None:
        self.buffer.push(*transition)

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        state = torch.as_tensor(batch["state"], dtype=torch.float32)
        action = torch.as_tensor(batch["action"], dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(1)
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
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_actor.load_state_dict(state_dict["target_actor"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
