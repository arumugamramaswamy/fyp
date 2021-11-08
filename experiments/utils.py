import supersuit as ss
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper


def wrap_parallel_env(parallel_env, num_envs=1):

    env = ss.pettingzoo_env_to_vec_env_v0(parallel_env)
    env = ss.concat_vec_envs_v0(env, num_envs, base_class="stable_baselines3")
    return env


def benchmark_random_agent(env: SB3VecEnvWrapper, N=200):

    all_rewards = []
    for ep_id in range(N):

        total_reward = 0
        obs = env.reset()
        while True:
            action = [env.action_space.sample() for _ in range(env.num_envs)]
            obs, rewards, dones, info = env.step(action)

            total_reward += rewards.sum()
            if dones.all():
                break

        all_rewards.append(total_reward)

    return all_rewards
