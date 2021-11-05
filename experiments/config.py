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
