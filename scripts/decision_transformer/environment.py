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

class ConfigurableActions(gs.MultiDiscrete):
    def __init__(self, g2op_env,
            modifying_bus_keys=None,
            modifying_bus_count=4,
            modifying_line_keys=None,
            modifying_line_count=4,
            curtail_bin_counts=11,
            redispatch_bin_counts=11
        ):

        self.template_action = g2op_env.action_space()
        multidiscrete_space = []

        self.num_buses = g2op_env.dim_topo
        self.num_lines = g2op_env.n_line
        self.num_generators = g2op_env.n_gen
        self.modifiable_buses = modifying_bus_count
        self.modifiable_lines = modifying_line_count

        if modifying_bus_keys is None:
            self.bus_keys = [i for i in range(1, g2op_env.dim_topo + 1)]
        else:
            self.bus_keys = modifying_bus_keys
        multidiscrete_space += [len(self.bus_keys) + 1] * self.modifiable_buses # +1 to add a no action value

        if modifying_line_keys is None:
            self.line_keys = [i for i in range(1, g2op_env.n_line)]
        else:
            self.line_keys = modifying_line_keys
        multidiscrete_space += [len(self.line_keys) + 1] * self.modifiable_lines # +1 to add a no action value

        self.num_curtail_bins = curtail_bin_counts
        self.curtail_mask = g2op_env.gen_renewable
        if curtail_bin_counts != 0:
            self.curtail_bins = np.linspace(
                start=0., # these numbers are pulled from the limits on the "curtailment" observation
                stop=1.,
                num=curtail_bin_counts
            )
            multidiscrete_space += [curtail_bin_counts + 1] * self.num_generators

        self.redispatch_bins = []
        self.num_redispatch_bins = redispatch_bin_counts
        self.dispatch_mask = g2op_env.gen_redispatchable
        max_dispatches = g2op_env.gen_max_ramp_up
        min_dispatches = g2op_env.gen_max_ramp_down
        if redispatch_bin_counts != 0:
            for i in range(g2op_env.n_gen):
                self.redispatch_bins.append(np.linspace(
                    start=min_dispatches[i],
                    stop=max_dispatches[i],
                    num=redispatch_bin_counts
                ))
                multidiscrete_space += [redispatch_bin_counts + 1]

        self.space_dim = len(multidiscrete_space)
        super().__init__(multidiscrete_space)

    def close(self):
        pass # stole this from grid2op as a reference
        # See the below link and scroll to the close method of the __AuxBoxGymActSpace class
        # https://github.com/Grid2op/grid2op/blob/master/grid2op/gym_compat/box_gym_actspace.py

    def no_action(self):
        return np.zeros(self.space_dim, dtype=np.int32)

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
            self,
            modifying_bus_keys=None,
            modifying_bus_count=4,
            modifying_line_keys=None,
            modifying_line_count=4,
            curtail_bin_counts=11,
            redispatch_bin_counts=11
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
        self.setup_actions(
            modifying_bus_keys, modifying_bus_count,
            modifying_line_keys, modifying_line_count,
            curtail_bin_counts, redispatch_bin_counts
        )

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

    def setup_observations(self):
        keeping_attributes = [
            "gen_p", "gen_q", "gen_v", "gen_theta", "gen_p_before_curtail",
            "gen_margin_up", "gen_margin_down",
            "target_dispatch", "actual_dispatch",
            "load_p", "load_q", "load_v", "load_theta",
            "rho", "thermal_limit",
            "line_status", "timestep_overflow",
            "p_or", "q_or", "v_or", "theta_or", "a_or",
            "p_ex", "q_ex", "v_ex", "theta_ex", "a_ex"
        ]
        self._gym_env.observation_space = gym_compat.BoxGymObsSpace(
            self._g2op_env.observation_space,
            attr_to_keep=keeping_attributes
        )

    def setup_actions(self, 
            modifying_bus_keys=None,
            modifying_bus_count=4,
            modifying_line_keys=None,
            modifying_line_count=4,
            curtail_bin_counts=11,
            redispatch_bin_counts=11
        ):
        """
            You will encounter a large amount of actions that will straight up just kill the grid, this is fine 
            To overcome it, the model needs to learn what will and wont kill the grid
        """
        
        self._gym_env.action_space = ConfigurableActions(self._g2op_env,
            modifying_bus_keys, modifying_bus_count,
            modifying_line_keys, modifying_line_count,
            curtail_bin_counts, redispatch_bin_counts
        )

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        return self._gym_env.render()