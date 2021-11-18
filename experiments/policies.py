import typing as T

import gym
import torch

from stable_baselines3.common.policies import ActorCriticPolicy


class ShufflePolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule,
        num_agents=1,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.num_agents = num_agents

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        landmarks = obs[:, 4 : self.num_agents * 2 + 4]
        landmarks = landmarks.reshape((obs.shape[0], self.num_agents, 2))

        for ind in range(len(landmarks)):
            perm = torch.randperm(self.num_agents)
            landmarks[ind, :] = landmarks[ind, perm, :]

        shuffled_landmarks = landmarks.reshape((obs.shape[0], self.num_agents * 2))

        other_agents = obs[
            :,
            4
            + self.num_agents * 2 : 4
            + self.num_agents * 2
            + (self.num_agents - 1) * 2,
        ]
        other_agents = other_agents.reshape((obs.shape[0], self.num_agents - 1, 2))

        for ind in range(len(other_agents)):
            perm = torch.randperm(self.num_agents - 1)
            other_agents[ind, :] = other_agents[ind, perm, :]

        shuffled_neighbours = other_agents.reshape(
            (obs.shape[0], (self.num_agents - 1) * 2)
        )

        obs = torch.cat(
            [
                obs[:, :4],
                shuffled_landmarks,
                shuffled_neighbours,
                obs[:, self.num_agents * 2 + (self.num_agents - 1) * 2 + 4 :],
            ],
            dim=-1,
        )
        return super().forward(obs, deterministic=deterministic)
