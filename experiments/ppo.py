import pathlib
from pettingzoo.mpe import simple_v2, simple_spread_v2
import time
import os
import typing as T
import torch as th
import gym

from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from experiments.utils import wrap_parallel_env


class CustomPolicy(ActorCriticPolicy):
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
        self, obs: th.Tensor, deterministic: bool = False
    ) -> T.Tuple[th.Tensor, th.Tensor, th.Tensor]:

        landmarks = obs[:, 4 : self.num_agents * 2 + 4]
        landmarks = landmarks.reshape((obs.shape[0], self.num_agents, 2))

        for ind in range(len(landmarks)):
            perm = th.randperm(self.num_agents)
            landmarks[ind, :] = landmarks[ind, perm, :]

        shuffled_landmarks = landmarks.reshape((obs.shape[0], self.num_agents * 2))
        obs = th.cat(
            [obs[:, :4], shuffled_landmarks, obs[:, self.num_agents * 2 + 4 :]], dim=-1
        )
        return super().forward(obs, deterministic=deterministic)


def create_simple_envs(
    train_episode_length, test_episode_length, num_training_envs
) -> T.Tuple[SB3VecEnvWrapper, SB3VecEnvWrapper]:
    env = simple_v2.parallel_env(max_cycles=train_episode_length)
    env = wrap_parallel_env(env, num_envs=num_training_envs)

    eval_env = simple_v2.parallel_env(max_cycles=test_episode_length)
    eval_env = wrap_parallel_env(eval_env, num_envs=1)
    return env, eval_env


def create_simple_spread_envs(
    train_episode_length, test_episode_length, num_training_envs, N=1
) -> T.Tuple[SB3VecEnvWrapper, SB3VecEnvWrapper]:
    """Create and wrap a simple spread environment.

    Args:
        N: number of agents in the simple spread env
    """
    env = simple_spread_v2.parallel_env(N=N, max_cycles=train_episode_length)
    env = wrap_parallel_env(env, num_envs=num_training_envs)

    eval_env = simple_spread_v2.parallel_env(N=N, max_cycles=test_episode_length)
    eval_env = wrap_parallel_env(eval_env, num_envs=1)
    return env, eval_env


def train(
    experiment_name: str,
    ppo_kwargs: T.Dict[str, T.Any],
    eval_callback_kwargs: T.Dict[str, T.Any],
    timesteps: int,
):
    """Train a given configuratio

    Args:
        experiment_name: name of the experiment
        ppo_kwargs: kwarg dictionary to be passed to the PPO algorithm
        eval_callback_kwargs: kwarg dictionary to be passed to the eval callback
        timestep: number of timesteps to train the algorithm for
    """

    # assign names of directories to use for saving
    dir_name = f"{experiment_name}-{datetime.now().strftime('%d-%m-%yT%H-%M-%S')}"
    model_save_path = os.path.join(dir_name, experiment_name)
    log_dir = os.path.join(dir_name, "logs/")
    pathlib.Path(dir_name).mkdir(exist_ok=True)
    eval_callback_kwargs.update(
        best_model_save_path=log_dir,
        log_path=log_dir,
    )

    # create model and callback
    model = PPO(**ppo_kwargs)
    eval_cb = EvalCallback(**eval_callback_kwargs)

    # train the model
    model.learn(total_timesteps=timesteps, callback=eval_cb)

    # save the trained model
    model.save(model_save_path)

    # save the config
    with open(model_save_path, "w") as f:
        f.write(
            "\n".join(
                [
                    "Experiment name: " + experiment_name,
                    "Model key word arguments: " + repr(ppo_kwargs),
                    "Eval callback key word arguments: " + repr(eval_callback_kwargs),
                    "Timesteps: " + repr(timesteps),
                ]
            )
        )

    return model, model_save_path


def test(
    env: SB3VecEnvWrapper,
    model: PPO,
    render=False,
    num_episodes=200,
):

    all_rewards = []
    for _ in range(num_episodes):

        total_reward = 0
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = env.step(action)

            if render:
                env.venv.vec_envs[0].par_env.unwrapped.render()
                time.sleep(0.1)

            total_reward += rewards.sum()
            if dones.all():
                break

        all_rewards.append(total_reward)

    return all_rewards
