from pettingzoo.mpe import simple_v2
import time

from stable_baselines3 import PPO
from torch.nn import ReLU

from utils import wrap_parallel_env

# Parallel environments
env = simple_v2.parallel_env(max_cycles=25)
env = wrap_parallel_env(env, num_envs=16)

print(f"Observation space: {env.observation_space}")
print(f"Number of environments: {env.num_envs}")

model = PPO("MlpPolicy", env, verbose=1 , policy_kwargs=dict(net_arch=[16,12,8], activation_fn=ReLU))
model.learn(total_timesteps=500000)
model.save("ppo_spread")

env = simple_v2.parallel_env(max_cycles=25)
env = wrap_parallel_env(env, num_envs=1)

model = PPO.load("ppo_spread") 
obs = env.reset() 
print(dir(env.venv.vec_envs[0].par_env.unwrapped.world.entities[0].state.p_pos)) 
env.venv.vec_envs[0].par_env.unwrapped.world.entities[0].state.p_pos[0] = 10 
env.venv.vec_envs[0].par_env.unwrapped.world.entities[0].state.p_pos[1] = 10 
while True: 
    action, _states = model.predict(obs) 
    obs, rewards, dones, info = env.step(action) 
    env.render() 
    time.sleep(0.1) 
    print(rewards)