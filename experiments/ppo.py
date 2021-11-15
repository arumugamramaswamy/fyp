import pathlib
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




def train(
    experiment_name: str,
    ppo_kwargs: T.Dict[str, T.Any],
    eval_callback_kwargs: T.Dict[str, T.Any],
    timesteps: int,
):
    """Train a given configuration

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
    ppo_kwargs.update(tensorboard_log=log_dir)

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
