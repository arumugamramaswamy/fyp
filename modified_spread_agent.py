import numpy as np
import typing as T

Observation = np.ndarray
Neighbourhood = T.List[Observation]
Action = T.Any
Value = T.Any
LogProb = T.Any


class SwarmAgent:
    def __init__(
        self, policy: T.Callable[[Observation, Neighbourhood], T.Tuple[Action, Value, LogProb]]
    ) -> None:
        self.policy = policy
        pass

    def step(self, observation):
        pass

    def _split_observation(self, observation):
        pass

    def _find_neighbourhood(self, candidate_neighbourhood):
        pass

    def learn(self, **kwargs):
        pass


class SwarmSpreadAgent(SwarmAgent):
    def __init__(self, policy, N, bound) -> None:
        super().__init__(policy=policy)
        self.N = N
        self.bound_sq = bound ** 2

    def _split_observation(self, observation):
        global_observations = observation[: 4 + 2 * self.N]
        candidate_neighbourhood_observations = observation[4 + 2 * self.N :]
        return global_observations, candidate_neighbourhood_observations

    def step(self, observation):
        (
            global_observations,
            candidate_neighbourhood_observations,
        ) = self._split_observation(observation)

        neighbourhood_observations = self._find_neighbourhood(
            candidate_neighbourhood_observations
        )
        assert len(global_observations) == 4 + self.N * 2
        assert len(candidate_neighbourhood_observations) == 4 * self.N - 4

        action, _, _ = self.policy(global_observations, neighbourhood_observations)
        return action

    def _find_neighbourhood(self, candidate_neighbourhood_observations):
        num_positions = 2 * self.N - 2
        pos = candidate_neighbourhood_observations[:num_positions]
        vel = candidate_neighbourhood_observations[num_positions:]

        neighbourhood = []
        for x in range(self.N - 1):
            curr_pos = pos[2 * x : 2 * x + 2]
            if (curr_pos ** 2).sum() < self.bound_sq:
                neighbourhood.append(np.concatenate([curr_pos, vel[2 * x : 2 * x + 2]]))

        return neighbourhood

class SwarmMeanEmbeddingPolicy:
    pass