import pathlib
from pettingzoo.mpe import simple_v2
import time
import os
import typing as T

from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from torch.nn import ReLU
from experiments.config import ExperimentConfig

from experiments.utils import wrap_parallel_env


def train(config: ExperimentConfig):

    dir_name = (
        f"{config.experiment_name}-{datetime.now().strftime('%d-%m-%yT%H-%M-%S')}"
    )
    model_save_path = os.path.join(dir_name, config.experiment_name)
    log_dir = os.path.join(dir_name, "logs/")

    pathlib.Path(dir_name).mkdir(exist_ok=True)

    # Parallel environments
    env = simple_v2.parallel_env(max_cycles=config.train_episode_length)
    env = wrap_parallel_env(env, num_envs=config.num_training_envs)

    eval_env = simple_v2.parallel_env(max_cycles=config.test_episode_length)
    eval_env = wrap_parallel_env(eval_env, num_envs=1)

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
        "MlpPolicy",
        env,
        batch_size=config.batch_size,
        n_steps=config.n_steps,
        verbose=1,
        policy_kwargs=dict(
            net_arch=config.network_arch,
            activation_fn=config.activation_function,
        ),
    )
    model.learn(total_timesteps=config.timesteps, callback=eval_cb)

    model.save(model_save_path)
    with open(model_save_path, "w") as f:
        f.write(repr(config) + "\n")

    return model, model_save_path


def test(
    model: PPO,
    config: ExperimentConfig,
    model_save_path: T.Optional[str] = None,
    render=False,
    N=200,
):

    env = simple_v2.parallel_env(max_cycles=config.test_episode_length)
    env = wrap_parallel_env(env, num_envs=1)

    all_rewards = []
    for ep_id in range(N):

        total_reward = 0
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
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
