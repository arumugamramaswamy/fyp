import typing as T
from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class ExperimentConfig:

    experiment_name: str
    network_arch: T.List[int]
    activation_function: T.Any
    num_training_envs: int
    timesteps: int
    batch_size: int = 64
    train_episode_length: int = 25
    test_episode_length: int = 100
    n_steps: int = 25
