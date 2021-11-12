env = "Placeholder"
eval_env = "Placeholder"
ReLU = "Placeholder"
example_train_config = dict(
    experiment_name = "ppo_simple",
    ppo_kwargs = dict(
        policy = "MlpPolicy",
        env=env,
        batch_size=64,
        verbose=1,
        policy_kwargs = dict(
            net_arch = [16,12,8],
            activation_fn=ReLU,
        ),
        n_epochs=4,
        n_steps=25
    ),
    eval_callback_kwargs = dict(
        eval_env=eval_env,
        deterministic=True,
        render=False,
        n_eval_episodes=100,
        eval_freq=(10000//16)
    ),
    timesteps = 100_000
)
