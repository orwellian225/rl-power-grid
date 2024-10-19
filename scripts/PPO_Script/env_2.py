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


class reducedActionSpace(gs.MultiDiscrete):
    def __init__(self,
                 env,
                 substation_key = [1,4,7,9,16,21,23,26,28,29,32,33],
                 curtail_bin_counts=11,
                 redispatch_bin_counts=11
                 ):

        self.template_action = env.action_space()
        self.substation_key = substation_key
        self.num_generators = env.n_gen
        # 1st value is sub-1,4,7,9,16,23,26,28,33
        space = [4] * len(substation_key)
        self.num_curtail_bins = curtail_bin_counts
        self.curtail_mask = env.gen_renewable
        if curtail_bin_counts != 0:
            self.curtail_bins = np.linspace(
                start=0.,  # these numbers are pulled from the limits on the "curtailment" observation
                stop=1.,
                num=curtail_bin_counts
            )
            space += [curtail_bin_counts + 1] * self.num_generators

        self.redispatch_bins = []
        self.num_redispatch_bins = redispatch_bin_counts
        self.dispatch_mask = env.gen_redispatchable
        max_dispatches = env.gen_max_ramp_up
        min_dispatches = env.gen_max_ramp_down
        if redispatch_bin_counts != 0:
            for i in range(env.n_gen):
                self.redispatch_bins.append(np.linspace(
                    start=min_dispatches[i],
                    stop=max_dispatches[i],
                    num=redispatch_bin_counts
                ))
                space += [redispatch_bin_counts + 1]

        super().__init__(space)

    def close(self):
        pass  # stole this from grid2op as a reference
        # See the below link and scroll to the close method of the __AuxBoxGymActSpace class
        # https://github.com/Grid2op/grid2op/blob/master/grid2op/gym_compat/box_gym_actspace.py

    def from_gym(self, gym_action):
        g2op_action = self.template_action.copy()
        for i in range(1,len(self.substation_key)):
            if gym_action[i] != 1:
                # gym_action[i] == 0 if you want to disconnect substation bus
                # if gym_action[i] == 1 do nothing
                # if gym_action[i] == 2 connect substation to bus 1
                # if gym_action[i] == 3 connect substation to bus 2
                g2op_action.set_bus = [(self.substation_key[i], gym_action[i]-1)]

        if self.num_curtail_bins != 0:
            start_curtail = len(self.substation_key)
            curtails = []
            for i in range(self.num_generators):
                if self.curtail_mask[i] and gym_action[start_curtail + i] != 0:
                    curtails.append(
                        (i, self.curtail_bins[gym_action[start_curtail + i] - 1])
                    )

            g2op_action.curtail = curtails

        if self.num_redispatch_bins != 0:
            start_redispatch = start_curtail + self.num_generators
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
        self._gym_env.action_space = reducedActionSpace(self._g2op_env, curtail_bin_counts=5, redispatch_bin_counts=5)

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        return self._gym_env.render()