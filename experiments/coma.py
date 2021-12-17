import torch
import typing as T

from torch import nn
from torch import optim
from torch.nn import functional
from torch.distributions import Categorical

from pettingzoo.mpe import simple_spread_v2


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, activation=nn.LeakyReLU):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        layers.append(activation())
        for x in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[x], hidden_sizes[x+1]))
            layers.append(activation())
        layers.append(nn.Linear(hidden_sizes[-1], output_dim))

        self.seq = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.seq(inputs)


class COMADiscreteActionCritic(nn.Module):
    """A critic based on the paper counterfactual multiagent policy gradient.
    
    This critic is being designed for training swarms and hence we will assume 
    that all the agents are similar: i.e. they have the same action space

    Design considerations:
        current architecture
            input to neural network: state + actions of all agents (one hot encoded?)
                + agent number to consider(one hot encoded?)
            output of neural network: action values for all actions for the given agent

        alternative architecute
            input to neural network: state + actions of all agents
            output of neural network: action values for all actions for all agents

    Args:
        state_dim: the shape of the state provided by the environment
        num_agents: the number of agents in the environment
        act_dim: the number of actions that can be taken by a given agent
        hidden_sizes: the size of each hidden layer
    """

    def __init__(self, state_dim, num_agents, act_dim, hidden_sizes):
        super().__init__()


        self.state_dim = state_dim
        self.num_agents = num_agents
        self.act_dim = act_dim

        input_shape = state_dim + num_agents + 1 # state dimenions + actions of each agent + agent id

        self.q = MLP(input_shape, act_dim, hidden_sizes)

    def forward(self, state, agent_actions, agent_id):

        x = torch.cat([state, agent_actions, agent_id],dim = -1)
        return self.q(x)

class Actor(nn.Module):
    """An actor based on the COMA paper"""

    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.pi = MLP(obs_dim, act_dim, hidden_sizes)

    def forward(self, obs):
        logits = self.pi(obs)
        distribution = Categorical(logits = logits)
        return distribution

def sample_action(distribution: Categorical, deterministic: bool = False):
    if deterministic:
        action = distribution.probs.argmax(dim=-1)
    else:
        action = distribution.sample()
    log_prob = distribution.log_prob(action)

    return action, log_prob
        
def calculate_advantage(q_values, action, probabilities):
    q_chosen_action = q_values.gather(-1, action)
    expected_values = q_values*probabilities
    return q_chosen_action - expected_values.sum()
    
def calculate_critic_loss(old_q_values, old_agent_actions, reward, new_q_values, new_agent_actions, gamma=0.99):
    q_old_state_action = old_q_values.gather(-1, old_agent_actions)
    q_new_state_action = new_q_values.gather(-1, new_agent_actions)

    td_target = reward + q_new_state_action*gamma

    return functional.mse_loss(q_old_state_action, td_target)

def calculate_actor_loss(log_prob, advantage):
    return (-log_prob*advantage).sum()
    

class COMA:

    def __init__(self, env, actor_kwargs: T.Dict[str, T.Any], critic_kwargs: T.Dict[str, T.Any]) -> None:
        self.actor = Actor(**actor_kwargs)
        self.critic = COMADiscreteActionCritic(**critic_kwargs)
        self.env = env
        self.episode_storage_buffer = []

    def train(self, num_episodes):
        
        N = self.env.num_agents

        actor_optim = optim.Adam(self.actor.parameters())
        critic_optim = optim.Adam(self.critic.parameters())

        for ep_id in range(num_episodes):
            agent_id = -1

            actions = []
            observations = []
            rewards = []
            log_probs = []
            probabilities = []
            return_val = 0

            for agent in self.env.agent_iter():

                agent_id += 1
                agent_id %= N

                state = self.env.state()

                observation, reward, done, info = self.env.last() 
                return_val += reward

                if done:
                    action = None
                    action_copy = None
                    log_prob = None
                    probability = None
                else:

                    # process observation and get action
                    observation = torch.from_numpy(observation)
                    distribution = self.actor(observation)
                    action, log_prob = sample_action(distribution)
                    action_copy = action
                    action = action.item()
                    probability = distribution.probs.detach()

                actions.append(action_copy)
                observations.append(observation)
                rewards.append(reward)
                log_probs.append(log_prob)
                probabilities.append(probability)

                if agent_id == N-1:

                    self.store_episode_step(state, observations, actions, rewards, log_probs, probabilities)
                    actions = []
                    observations = []
                    rewards = []
                    log_probs = []
                    probabilities = []

                self.env.step(action)
            self.env.reset()
            print(return_val)

            actor_optim.zero_grad()
            critic_optim.zero_grad()
            actor_loss = torch.tensor(0, dtype=torch.float)
            critic_loss = torch.tensor(0, dtype=torch.float)

            for step_id in range(len(self.episode_storage_buffer)-1):
                old_state, _, old_actions, _, old_log_probs, probabilities = self.episode_storage_buffer[step_id]
                new_state, _, new_actions, rewards, _, _ = self.episode_storage_buffer[step_id + 1]

                for x in range(N):
                    old_state_value = self.critic(torch.from_numpy(old_state), torch.Tensor(old_actions), torch.tensor([x]))

                    advantage_state_value = old_state_value.detach()
                    advantage = calculate_advantage(advantage_state_value, old_actions[x], probabilities[x])
                    actor_loss += calculate_actor_loss(old_log_probs[x], advantage)

                    if new_actions[x] is None:
                        continue

                    new_state_value = self.critic(torch.from_numpy(new_state), torch.Tensor(new_actions), torch.tensor([x])).detach()

                    critic_loss += calculate_critic_loss(
                        old_state_value,
                        torch.tensor(old_actions,dtype=torch.int64),
                        rewards[x],
                        new_state_value,
                        torch.tensor(new_actions,dtype=torch.int64)
                    )
                    
            actor_loss.backward()
            critic_loss.backward()

            actor_optim.step()
            critic_optim.step()

            # empty the storage buffer
            self.episode_storage_buffer = []

    def test(self, num_episodes):
        
        N = self.env.num_agents

        for ep_id in range(num_episodes):
            agent_id = -1

            return_val = 0

            for agent in self.env.agent_iter():

                agent_id += 1
                agent_id %= N

                if agent_id == 0:
                    self.env.render()

                observation, reward, done, info = self.env.last() 
                return_val += reward

                if done:
                    action = None
                else:

                    # process observation and get action
                    observation = torch.from_numpy(observation)
                    distribution = self.actor(observation)
                    action, log_prob = sample_action(distribution, True)
                    action = action.item()


                self.env.step(action)


            self.env.reset()
            print(return_val)

    def store_episode_step(self, state, observations, actions, rewards, log_probs, probability):
        self.episode_storage_buffer.append([state, observations, actions, rewards, log_probs, probability])

def main(N=3):
    env = simple_spread_v2.env(N=N, max_cycles=200)
    env.env.env.world.dim_c = 0
    num_episodes = 1000

    env.reset()

    act_dim = 5
    obs_dim = env.last()[0].shape[0]
    
    coma = COMA(env, dict(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=[64]
    ), dict(
        state_dim = env.state().shape[0],
        num_agents=N,
        act_dim=act_dim,
        hidden_sizes=[64]
    ))
    coma.train(num_episodes)
    print("====testing====")
    coma.test(5)
    
    return coma

if __name__ == "__main__":
    coma = main(2)

