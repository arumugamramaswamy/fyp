from experiments import registry
import typing as T

def ppo_config_parser(config: T.Dict[str, T.Any]):
    env_name = config['env']['name']
    env_kwargs = config['env']['kwargs']

    create_env_fn = registry.ENV_REGISTRY[env_name]
    env, eval_env = create_env_fn(**env_kwargs)

    algo_name = config['train']['name']
    train_kwargs = config['train']['kwargs']

    train_fn = registry.ALGORITHM_REGISTRY[algo_name]['train']
    test_fn = registry.ALGORITHM_REGISTRY[algo_name]['test']

    # update policy
    policy_name = train_kwargs['ppo_kwargs']['policy']
    policy = registry.POLICY_REGISTRY[policy_name]
    train_kwargs['ppo_kwargs']['policy'] = policy

    # update envs
    train_kwargs['ppo_kwargs']['env'] = env
    train_kwargs['eval_callback_kwargs']['eval_env'] = eval_env

    # update activation_fn
    if 'policy_kwargs' in train_kwargs['ppo_kwargs']:
        if 'activation_fn' in train_kwargs['ppo_kwargs']['policy_kwargs']:

            activation_fn_name = train_kwargs['ppo_kwargs']['policy_kwargs']['activation_fn']
            activation_fn = registry.ACTIVATION_FN_REGISTRY[activation_fn_name]

            train_kwargs['ppo_kwargs']['policy_kwargs']['activation_fn'] = activation_fn

    train = lambda: train_fn(**train_kwargs)
    test = lambda model: test_fn(env=eval_env, model=model)
    return train, test
