from experiments import registry
import typing as T
import copy
import pathlib
import os.path
import yaml


def _load_and_merge_base_config(config: T.Dict[str, T.Any], config_path):
    if "base" not in config:
        return config

    parent_path = pathlib.Path(config_path).parent
    relative_base_path: str = config["base"]
    relative_path = parent_path.joinpath(relative_base_path)
    base_config_path = relative_path.resolve().as_posix()

    with open(base_config_path, "r") as f:
        base_config = yaml.load(f, Loader=yaml.Loader)

    return _merge_dicts(base_config, config)

def _merge_dicts(base_dict: T.Dict, update_dict: T.Dict):

    new_dict = copy.deepcopy(base_dict)
    for key, value in update_dict.items():
        if (
            isinstance(value, dict)
            and key in base_dict
            and isinstance(base_dict[key], dict)
        ):
            new_dict[key] = _merge_dicts(base_dict[key], value)
            continue

        new_dict[key] = value
    return new_dict

def ppo_config_parser(config: T.Dict[str, T.Any], config_path):

    config = _load_and_merge_base_config(config, config_path)
    raw_config = copy.deepcopy(config)

    print("Parsing config... ")
    print(config)

    env_name = config["env"]["name"]
    env_kwargs = config["env"]["kwargs"]

    create_env_fn = registry.ENV_REGISTRY[env_name]
    env, eval_env = create_env_fn(**env_kwargs)

    algo_name = config["train"]["name"]
    train_kwargs = config["train"]["kwargs"]

    train_fn = registry.ALGORITHM_REGISTRY[algo_name]["train"]
    test_fn = registry.ALGORITHM_REGISTRY[algo_name]["test"]

    # update policy
    policy_name = train_kwargs["ppo_kwargs"]["policy"]
    policy = registry.POLICY_REGISTRY[policy_name]
    train_kwargs["ppo_kwargs"]["policy"] = policy

    # update envs
    train_kwargs["ppo_kwargs"]["env"] = env
    train_kwargs["eval_callback_kwargs"]["eval_env"] = eval_env

    # update activation_fn
    if "policy_kwargs" in train_kwargs["ppo_kwargs"]:
        if "activation_fn" in train_kwargs["ppo_kwargs"]["policy_kwargs"]:

            activation_fn_name = train_kwargs["ppo_kwargs"]["policy_kwargs"][
                "activation_fn"
            ]
            activation_fn = registry.ACTIVATION_FN_REGISTRY[activation_fn_name]

            train_kwargs["ppo_kwargs"]["policy_kwargs"]["activation_fn"] = activation_fn

    train = lambda: train_fn(**train_kwargs)
    test = lambda model: test_fn(env=eval_env, model=model)
    return train, test, raw_config
