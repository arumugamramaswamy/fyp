from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from modified_spread_scenario import Scenario

class raw_env(SimpleEnv):
    def __init__(self, N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False):
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles, continuous_actions, local_ratio)
        self.metadata["name"] = "modified_spread"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
