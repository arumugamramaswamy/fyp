import experiments.ppo
import experiments.env
import experiments.policies
import torch

ALGORITHM_REGISTRY = {
    "ppo": {
        "train": experiments.ppo.train,
        "test": experiments.ppo.test
    }
}

ENV_REGISTRY = {
    "simple": experiments.env.create_simple_envs,
    "simple_spread": experiments.env.create_simple_spread_envs
}

ACTIVATION_FN_REGISTRY = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "elu": torch.nn.ELU,
}

POLICY_REGISTRY = {
    "mlp": "MlpPolicy",
    "shuffle": experiments.policies.ShufflePolicy
}
