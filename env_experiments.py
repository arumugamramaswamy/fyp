from modified_spread_agent import SwarmSpreadAgent
import modified_spread_env
env = modified_spread_env.env(N=5, max_cycles=500)

mpe_agent = SwarmSpreadAgent(lambda *args: ..., 5,1)
env.reset()
for id, agent in enumerate(env.agent_iter()):
    observation, reward, done, info = env.last()
    action = env.action_spaces[agent].sample()
    print(observation, agent, reward, done, info, action)

    if not env.render():
        continue
    env.step(action)