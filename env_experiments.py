
import modified_spread_env
env = modified_spread_env.env(N=5, max_cycles=500)

env.reset()
for id, agent in enumerate(env.agent_iter()):
    observation, reward, done, info = env.last()
    action = env.action_spaces[agent].sample()
    print(observation, agent, reward, done, info, action)
    print(observation.shape)
    env.render()
    env.step(action)