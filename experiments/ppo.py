import pathlib
from pettingzoo.mpe import simple_v2, simple_spread_v2
import time
import os
import typing as T
import torch as th
import gym

from torch import nn

from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
from torch.nn import ReLU
from experiments.config import ExperimentConfig

from experiments.utils import wrap_parallel_env


class CustomPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        num_agents=1,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.num_agents = num_agents

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> T.Tuple[th.Tensor, th.Tensor, th.Tensor]:

        landmarks = obs[:, 4 : self.num_agents * 2 + 4]
        landmarks = landmarks.reshape((obs.shape[0], self.num_agents, 2))

        for ind, landmark in enumerate(landmarks):
            perm = th.randperm(self.num_agents)
            landmarks[ind, :] = landmarks[ind, perm, :]


        shuffled_landmarks = landmarks.reshape((obs.shape[0], self.num_agents * 2))
        obs = th.cat([obs[:, :4], shuffled_landmarks, obs[:, self.num_agents * 2 + 4 :]], dim=-1)
        return super().forward(obs, deterministic=deterministic)


def create_simple_envs(
    config: ExperimentConfig,
) -> T.Tuple[SB3VecEnvWrapper, SB3VecEnvWrapper]:
    env = simple_v2.parallel_env(max_cycles=config.train_episode_length)
    env = wrap_parallel_env(env, num_envs=config.num_training_envs)

    eval_env = simple_v2.parallel_env(max_cycles=config.test_episode_length)
    eval_env = wrap_parallel_env(eval_env, num_envs=1)
    return env, eval_env


def create_simple_spread_envs(
    config: ExperimentConfig, N=1
) -> T.Tuple[SB3VecEnvWrapper, SB3VecEnvWrapper]:
    env = simple_spread_v2.parallel_env(N=N, max_cycles=config.train_episode_length)
    env = wrap_parallel_env(env, num_envs=config.num_training_envs)

    eval_env = simple_spread_v2.parallel_env(N=N, max_cycles=config.test_episode_length)
    eval_env = wrap_parallel_env(eval_env, num_envs=1)
    return env, eval_env


def train(env: SB3VecEnvWrapper, eval_env: SB3VecEnvWrapper, config: ExperimentConfig):

    dir_name = (
        f"{config.experiment_name}-{datetime.now().strftime('%d-%m-%yT%H-%M-%S')}"
    )
    model_save_path = os.path.join(dir_name, config.experiment_name)
    log_dir = os.path.join(dir_name, "logs/")

    pathlib.Path(dir_name).mkdir(exist_ok=True)

    # Parallel environments

    eval_freq = 10_000
    eval_freq = max(eval_freq // config.num_training_envs, 1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        deterministic=True,
        render=False,
        eval_freq=eval_freq,
        n_eval_episodes=100,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Number of environments: {env.num_envs}")

    model = PPO(
        CustomPolicy,
        env,
        batch_size=config.batch_size,
        n_steps=config.n_steps,
        verbose=1,
        policy_kwargs=dict(
            net_arch=config.network_arch,
            activation_fn=config.activation_function,
            num_agents=2,
        ),
        n_epochs=4
    )
    model.learn(total_timesteps=config.timesteps, callback=eval_cb)

    model.save(model_save_path)
    with open(model_save_path, "w") as f:
        f.write(repr(config) + "\n")

    return model, model_save_path


def test(
    env: SB3VecEnvWrapper,
    model: PPO,
    config: ExperimentConfig,
    model_save_path: T.Optional[str] = None,
    render=False,
    N=200,
):

    all_rewards = []
    for ep_id in range(N):

        total_reward = 0
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)

            if render:
                env.venv.vec_envs[0].par_env.unwrapped.render()
                time.sleep(0.1)

            total_reward += rewards.sum()
            if dones.all():
                break

        all_rewards.append(total_reward)

    if model_save_path is not None:
        with open(model_save_path, "a") as f:
            f.write(repr(all_rewards) + "\n")
            f.write(repr(sum(all_rewards) / len(all_rewards)) + "\n")

    return all_rewards
