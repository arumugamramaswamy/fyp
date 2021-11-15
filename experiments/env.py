import typing as T

from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
from pettingzoo.mpe import simple_v2, simple_spread_v2

from experiments.utils import wrap_parallel_env

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
