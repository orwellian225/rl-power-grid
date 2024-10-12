import gymnasium as gym
from gymnasium import spaces
import numpy as np

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend

from gymnasium.spaces import MultiBinary, MultiDiscrete, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

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

        # print(f"Action Class:  {action_class.alertable_line_names}")
        # print(f"Observation Class:  {observation_class}")
        # print(f"Reward Class:  {reward_class}")

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

        #self.setup_observations()


        self.observation_space = self._gym_env.observation_space
        #self.action_space = self._gym_env.action_space
        self.setup_actions()



    def setup_observations(self):
        # TODO: Your code to specify & modify the observation space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        # Fetch the original observation space from the Grid2Op environment
        # original_observation_space = self._gym_env.observation_space
        # Example: Focus on relevant features only
        # In this example, let's say we focus on:
        # - Power flow on lines
        # - Generator outputs
        # - Line status (operational or not)

        # # Original observation space could be a dict; we need to filter relevant elements
        # filtered_observation_space = spaces.Dict({
        #     # Add only the relevant observation elements
        #     'rho': original_observation_space['rho'],  # Power flow through the grid's lines
        #     'power_output': original_observation_space['p_or'],  # Generator output
        #     'load power': original_observation_space['load_p'],  # Whether lines are operational
        # })

        # Update the observation space of the environment to use this filtered space
        # self.observation_space = filtered_observation_space

        print(f"Filtered observation space setup complete: {self.observation_space}")
        return self._gym_env.observation_space

        print("WARNING: setup_observations is not doing anything. Implement your own code in this method.")

    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        # Create a new simplified action space without curtailment or redispatch (the continuous ones)
        # Combine the discrete and binary actions into a single flattened MultiDiscrete action space
        self.action_space = MultiDiscrete(
            [2] * 57 + [2] * 20 + [3] * 57 + [3] * 20  # Flatten all actions into a single MultiDiscrete array
        )
        print("Flattened action space setup complete.")



        # print("WARNING: setup_actions is not doing anything. Implement your own code in this method.")

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        #return self._gym_env.step(action)
        # Unpack the flattened action into the original components
        change_bus_action = action[:57]
        change_line_status_action = action[57:77]
        set_bus_action = action[77:134]
        set_line_status_action = action[134:]

        # Combine these actions back into the original dictionary format expected by Grid2Op
        combined_action = {
            'change_bus': change_bus_action,
            'change_line_status': change_line_status_action,
            'set_bus': set_bus_action,
            'set_line_status': set_line_status_action
        }

        # Take a step in the environment with the combined action
        return self._gym_env.step(combined_action)

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()


def main():
    # Random agent interacting in environment #

    # Initialize the environment
    env = Gym2OpEnv()

    # Check the environment to make sure it's compatible with Stable Baselines3
    check_env(env, warn=True)
    # Create the PPO model with the appropriate policy for a discrete action space
    model = PPO("MultiInputPolicy", env, verbose=1, n_epochs=10, batch_size=128, n_steps=1280)

    # Train the PPO agent for a set number of timesteps

    model.learn(total_timesteps=40000)

    # Save the trained model
    model.save("ppo_grid2op")

    # Test the trained model
    obs, _ = env.reset()
    ppo_r = 0
    for step in range(2000):
        action, _states = model.predict(obs, deterministic=True)
        # Unpack the five values from the step function
        obs, reward, terminated, truncated, info = env.step(action)
        ppo_r += reward
        #env.render()
        if terminated or truncated:
            obs, _ = env.reset()
            break

    print(f"PPO reward: {ppo_r}")
    print(f"Steps: {step}")
    print(f"PPO training and testing complete")


    #env.reset()

    random_r = 0
    #count_steps = 0
    for step in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        random_r += reward
        #count_steps += 1
        if terminated or truncated:
            #count_steps = step
            break

    print(f"Random reward: {random_r}")
    print(f"Steps: {step}")
    # max_steps = 1
    #
    # env = Gym2OpEnv()
    #
    # print("#####################")
    # print("# OBSERVATION SPACE #")
    # print("#####################")
    # print(env.observation_space)
    # print("#####################\n")
    #
    # print("#####################")
    # print("#   ACTION SPACE    #")
    # print("#####################")
    # print(env.action_space)
    # print("#####################\n\n")
    #
    # curr_step = 0
    # curr_return = 0
    #
    # is_done = False
    # obs, info = env.reset()
    # print(f"step = {curr_step} (reset):")
    # print(f"\t obs = {obs}")
    # print(f"\t info = {info}\n\n")
    #
    # while not is_done and curr_step < max_steps:
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #
    #     curr_step += 1
    #     curr_return += reward
    #     is_done = terminated or truncated
    #
    #     print(f"step = {curr_step}: ")
    #     print(f"\t obs = {obs}")
    #     print(f"\t reward = {reward}")
    #     print(f"\t terminated = {terminated}")
    #     print(f"\t truncated = {truncated}")
    #     print(f"\t info = {info}")
    #
    #     # Some actions are invalid (see: https://grid2op.readthedocs.io/en/latest/action.html#illegal-vs-ambiguous)
    #     # Invalid actions are replaced with 'do nothing' action
    #     is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
    #     print(f"\t is action valid = {is_action_valid}")
    #     if not is_action_valid:
    #         print(f"\t\t reason = {info['exception']}")
    #     print("\n")
    #
    # print("###########")
    # print("# SUMMARY #")
    # print("###########")
    # print(f"return = {curr_return}")
    # print(f"total steps = {curr_step}")
    # print("###########")






if __name__ == "__main__":
    main()
