from experiments.config_parser import ppo_config_parser


def test_ppo_parser():
    config = {
        "env": {
            "name": "simple",
            "kwargs": {
                "train_episode_length":50,
                "test_episode_length":200,
                "num_training_envs":16
            }
        },
        "train": {
            "name": "ppo",
            "kwargs": {
                "experiment_name": "ppo_simple",
                "ppo_kwargs" : {
                    "policy": "mlp",
                    "batch_size":64,
                    "verbose":1,
                    "policy_kwargs": {
                        "net_arch": [16,12,8],
                        "activation_fn":"relu",
                    },
                    "n_epochs":4,
                    "n_steps":25
                },
                "eval_callback_kwargs":{
                    "deterministic":True,
                    "render":False,
                    "n_eval_episodes":100,
                    "eval_freq":(10000//16)
                },
                "timesteps" : 100_000
            }
        }
    }

    ppo_config_parser(config, "")
