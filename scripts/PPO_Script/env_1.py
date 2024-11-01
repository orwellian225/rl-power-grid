import gymnasium as gym

import numpy as np

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend

import gymnasium.spaces as gs

"""
    Input parameters:
        * env: the grid2op environment
        * num_modifiable_bus_objs: The number of objects you want to potentially modify the bus of each timestep
        * num_modifiable_lines: The number of lines you want to potentially change the status of each timestep
        * num_curtail_bins: The number of bins to discretize the curtailment into for each generator
        * num_redispatch_bins: The number of bins to discretize the dispatch into for each generator

    The space:
        A multidiscrete array of integers
            * The first num_modifiable_bus_objs correspond to modifying one of the number of bus objects
            * The next num_modifiable_lines correpsond to modifying the line status of one of the lines
            * The next num_generators correspond to the curtailment bin - Note that some outputs will be masked out if they can't be curtails
            * The next num_generators correspond to the redispacth bin - Note that some outputs will be masked out if they can't be redispatched

"""
class FlatLegalActionSpace(gs.MultiDiscrete):   
    def __init__(self,
        env,
        num_modifiable_bus_objs, 
        num_modifiable_lines,
        num_curtail_bins,
        num_redispatch_bins
    ):

        self.template_action = env.action_space()

        self.num_buses = env.dim_topo
        self.num_lines = env.n_line
        self.num_generators = env.n_gen
        self.modifiable_buses = num_modifiable_bus_objs
        self.modifiable_lines = num_modifiable_lines

        space = [env.dim_topo + 1] * num_modifiable_bus_objs + [env.n_line + 1] * num_modifiable_lines
        self.num_curtail_bins = num_curtail_bins
        self.curtail_mask = env.gen_renewable
        if num_curtail_bins != 0:
            self.curtail_bins = np.linspace(
                start=0., # these numbers are pulled from the limits on the "curtailment" observation
                stop=1.,
                num=num_curtail_bins
            )
            space += [num_curtail_bins + 1] * self.num_generators

        self.redispatch_bins = []
        self.num_redispatch_bins = num_redispatch_bins
        self.dispatch_mask = env.gen_redispatchable
        max_dispatches = env.gen_max_ramp_up
        min_dispatches = env.gen_max_ramp_down
        if num_redispatch_bins != 0:
            for i in range(env.n_gen):
                self.redispatch_bins.append(np.linspace(
                    start=min_dispatches[i],
                    stop=max_dispatches[i],
                    num=num_redispatch_bins
                ))
                space += [num_redispatch_bins + 1]

        self.space_len = len(space)
        super().__init__(space)

    def no_action(self):
        return np.zeros(self.space_len, dtype=np.int64)

    def close(self):
        pass # stole this from grid2op as a reference
        # See the below link and scroll to the close method of the __AuxBoxGymActSpace class
        # https://github.com/Grid2op/grid2op/blob/master/grid2op/gym_compat/box_gym_actspace.py

    def from_gym(self, gym_action):
        g2op_action = self.template_action.copy()
        for bus_idx in gym_action[:self.modifiable_buses]:
            if bus_idx != 0:
                g2op_action.change_bus = [bus_idx - 1]

        for line_idx in gym_action[self.modifiable_buses:self.modifiable_buses + self.modifiable_lines]:
            if line_idx != 0:
                g2op_action.change_line_status = [line_idx - 1]

        if self.num_curtail_bins != 0:
            start_curtail = self.modifiable_buses + self.modifiable_lines
            curtails = []
            for i in range(self.num_generators):
                if self.curtail_mask[i] and gym_action[start_curtail + i] != 0:
                    curtails.append(
                        (i, self.curtail_bins[gym_action[start_curtail + i] - 1])
                    )

            g2op_action.curtail = curtails

        if self.num_redispatch_bins != 0:
            start_redispatch = self.modifiable_buses + self.modifiable_lines + self.num_generators
            dispatches = np.zeros(self.num_generators)
            for i in range(self.num_generators):
                if gym_action[start_redispatch + i - 1] != 0:
                    dispatches[i] = self.redispatch_bins[i][gym_action[start_redispatch + i] - 1]

            g2op_action.redispatch = dispatches * self.dispatch_mask 
            
        return g2op_action

# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(
            self
    ):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        ##########
        # REWARD #
        ##########
        # NOTE: This reward should not be modified when evaluating RL agent
        # See https://grid2op.readthedocs.io/en/latest/reward.html
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        # reward = N1 + L2RPN
        cr.initialize(self._g2op_env)
        ##########

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        self.setup_observations()
        self.setup_actions()

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

    def setup_observations(self):
        pass

    def setup_actions(self):
        """
            You will encounter a large amount of actions that will straight up just kill the grid, this is fine 
            To overcome it, the model needs to learn what will and wont kill the grid
        """
        
        self._gym_env.action_space = FlatLegalActionSpace(self._g2op_env, 1, 1, 0, 0)

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        return self._gym_env.render()


def main():
    # Random agent interacting in environment #

    max_steps = 100

    env = Gym2OpEnv()

    print("#####################")
    print("# OBSERVATION SPACE #")
    print("#####################")
    print(env.observation_space)
    print("#####################\n")

    print("#####################")
    print("#   ACTION SPACE    #")
    print("#####################")
    print(env.action_space)
    print("#####################\n\n")

    curr_step = 0
    curr_return = 0

    is_done = False
    obs, info = env.reset()
    print(f"step = {curr_step} (reset):")
    print(f"\t obs = {obs}")
    print(f"\t info = {info}\n\n")

    while not is_done and curr_step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        curr_step += 1
        curr_return += reward
        is_done = terminated or truncated

        print(f"step = {curr_step}: ")
        print(f"\t obs = {obs}")
        print(f"\t reward = {reward}")
        print(f"\t terminated = {terminated}")
        print(f"\t truncated = {truncated}")
        print(f"\t info = {info}")

        # Some actions are invalid (see: https://grid2op.readthedocs.io/en/latest/action.html#illegal-vs-ambiguous)
        # Invalid actions are replaced with 'do nothing' action
        is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
        print(f"\t is action valid = {is_action_valid}")
        if not is_action_valid:
            print(f"\t\t reason = {info['exception']}")
        print("\n")

    print("###########")
    print("# SUMMARY #")
    print("###########")
    print(f"return = {curr_return}")
    print(f"total steps = {curr_step}")
    print("###########")


if __name__ == "__main__":
    # main()
    pass
