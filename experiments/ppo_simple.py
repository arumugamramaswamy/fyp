import pathlib
from pettingzoo.mpe import simple_v2
import time
import os

from datetime import datetime
from stable_baselines3 import PPO
from torch.nn import ReLU
from experiments.config import ExperimentConfig

from experiments.utils import wrap_parallel_env


def train(config: ExperimentConfig):

    # Parallel environments
    env = simple_v2.parallel_env(max_cycles=25)
    env = wrap_parallel_env(env, num_envs=config.num_training_envs)

    print(f"Observation space: {env.observation_space}")
    print(f"Number of environments: {env.num_envs}")

    model = PPO(
        "MlpPolicy",
        env,
        batch_size=config.batch_size,
        verbose=1,
        policy_kwargs=dict(
            net_arch=config.network_arch,
            activation_fn=config.activation_function,
        ),
    )
    model.learn(total_timesteps=config.timesteps)

    dir_name = f"{config.experiment_name}-{datetime.now().strftime('%d-%m-%yT%H-%M-%S')}"
    model_save_path = os.path.join(dir_name, config.experiment_name)

    pathlib.Path(dir_name).mkdir(exist_ok=True)

    model.save(model_save_path)
    with open(model_save_path, "w") as f:
        f.write(repr(config))


def test():
    # TODO

    env = simple_v2.parallel_env(max_cycles=25)
    env = wrap_parallel_env(env, num_envs=1)

    model = PPO.load("ppo_spread")
    obs = env.reset()
    print(dir(env.venv.vec_envs[0].par_env.unwrapped.world.entities[0].state.p_pos))
    env.venv.vec_envs[0].par_env.unwrapped.world.entities[0].state.p_pos[0] = 10
    env.venv.vec_envs[0].par_env.unwrapped.world.entities[0].state.p_pos[1] = 10
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        time.sleep(0.1)
        print(rewards)
