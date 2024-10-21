import gymnasium as gym
import numpy as np
import pygame

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import DiscreteActSpace
from  grid2op.gym_compat import ContinuousToDiscreteConverter
from grid2op.gym_compat import ScalerAttrConverter

from lightsim2grid import LightSimBackend

from gymnasium.spaces import Discrete
from gymnasium.spaces import Box

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
        """
        Configure the observation space to include normalized values of:
        - rho: power flow divided by thermal limit for each line
        - gen_p: active power production normalized by max capacity
        - load_p: active power consumption normalized relative to baseline
        - topo_vect: current topology vector
        Plus additional custom observations like connectivity matrix.
        """
        # Start with full observation space
        obs_space = self._gym_env.observation_space
        
        # List of attributes to keep
        keep_attributes = [
            "rho",          # Power flow / thermal limit for each line
            "gen_p",        # Active power production
            "load_p",       # Active power consumption
            "topo_vect"     # Current topology vector
        ]
        
        # Get all attributes that we want to remove
        all_attributes = obs_space.spaces.keys()
        remove_attributes = [attr for attr in all_attributes if attr not in keep_attributes]
        
        # Remove unwanted attributes from observation space
        for attr in remove_attributes:
            self._gym_env.observation_space = self._gym_env.observation_space.ignore_attr(attr)
        
        # Get initial observation for scaling reference
        init_obs = self._g2op_env.reset()
        obs_gym = self._gym_env.observation_space.to_gym(init_obs)
        
        # Scale observation space using ScalerAttrConverter
        obs_space = self._gym_env.observation_space
        
        # Scale generator power by maximum capacity
        obs_space = obs_space.reencode_space(
            "gen_p",
            ScalerAttrConverter(
                substract=0.,  # Don't shift
                divide=self._g2op_env.gen_pmax  # Normalize by max capacity
            )
        )
        
        # Scale load power relative to initial load
        obs_space = obs_space.reencode_space(
            "load_p",
            ScalerAttrConverter(
                substract=obs_gym["load_p"],  # Center around initial load
                divide=0.5 * obs_gym["load_p"]  # Scale to roughly [-1, 1]
            )
        )
        
        # rho is already normalized by thermal limits
        # topo_vect is already in {1, 2}, but we can normalize it to [0, 1]
        obs_space = obs_space.reencode_space(
            "topo_vect",
            ScalerAttrConverter(
                substract=1.,  # Shift from {1,2} to {0,1}
                divide=1.
            )
        )
        
        # Custom observations
        # Add connectivity matrix
        shape_conn = (self._g2op_env.dim_topo, self._g2op_env.dim_topo)
        obs_space.add_key(
            "connectivity_matrix",
            lambda obs: obs.connectivity_matrix(),
            Box(
                shape=shape_conn,
                low=np.zeros(shape_conn),
                high=np.ones(shape_conn)
            )
        )
        
        # Add line capacity usage (percentage of thermal limit)
        obs_space.add_key(
            "line_capacity_usage",
            lambda obs: np.clip(obs.rho, 0., 1.),  # Clip to [0,1]
            Box(
                shape=(self._g2op_env.n_line,),
                low=np.zeros(self._g2op_env.n_line),
                high=np.ones(self._g2op_env.n_line)
            )
        )
        
        # Add binary indicator for line overflows
        obs_space.add_key(
            "line_overflows",
            lambda obs: obs.rho > 1.,  # Binary indicator for overloaded lines
            Box(
                shape=(self._g2op_env.n_line,),
                low=np.zeros(self._g2op_env.n_line),
                high=np.ones(self._g2op_env.n_line)
            )
        )
        
        self._gym_env.observation_space = obs_space

    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        # print("WARNING: setup_actions is not doing anything. Implement your own code in this method.")

        """
        Configure the action space for the Grid2Op environment.
        Focuses on change_bus, change_line_status, redispatch and curtailment actions.
        """
        # Start with full action space
        self._gym_env.action_space = self._gym_env.action_space

        # Remove set_bus and set_line_status to reduce complexity
        self._gym_env.action_space = self._gym_env.action_space.ignore_attr("set_bus").ignore_attr("set_line_status")
        
        # Convert continuous actions to discrete actions
        self._gym_env.action_space = self._gym_env.action_space.reencode_space(
            "curtail", 
            ContinuousToDiscreteConverter(nb_bins=3)
        )
        self._gym_env.action_space = self._gym_env.action_space.reencode_space(
            "redispatch", 
            ContinuousToDiscreteConverter(nb_bins=5)
        )
        
        # Convert to discrete action space
        self._gym_env.action_space = DiscreteActSpace(
            self._g2op_env.action_space,
            self._gym_env.action_space
        )
        
        # Store the number of actions but keep the Grid2Op action space
        self._action_space_n = self._gym_env.action_space.n

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()
    
    def close(self):
        """Properly close the environment."""
        if hasattr(self, '_original_act_space'):
            self._original_act_space.close()
        if hasattr(self, '_g2op_env'):
            self._g2op_env.close()
        if hasattr(self, '_gym_env'):
            self._gym_env.close()
        super().close()

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
    main()

	
