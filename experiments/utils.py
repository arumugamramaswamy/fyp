import supersuit as ss

def wrap_parallel_env(parallel_env, num_envs=1):
    
    env = ss.pettingzoo_env_to_vec_env_v0(parallel_env)
    env = ss.concat_vec_envs_v0(env, num_envs, base_class="stable_baselines3")
    return env