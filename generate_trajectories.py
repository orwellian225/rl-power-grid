from datetime import datetime
import pandas as pd
import numpy as np
import random

from scripts.env import Gym2OpEnv
from scripts.util import print_observation, get_formatted_date
import scripts.agents.decision_transformer.util as dtutil


"""
    # Diffusion Transformer Notes

    ## Environment

    [L2RPN Environment Description](https://grid2op.readthedocs.io/en/latest/available_envs.html)

        * No maintenance
        * No opponent 
        * No storage

    ## Data Generation

        1. Random Trajectory
"""

np.set_printoptions(linewidth=180)

env = Gym2OpEnv()

def generate_full_random_trajectories(num_trajectories, max_timesteps):
    generate_freq_random_trajectories(num_trajectories, max_timesteps, 1.0)

def generate_freq_random_trajectories(num_trajectories, max_timesteps, frequency):
    print(f"Trajectories are at most {max_timesteps} timesteps in length")
    for r in range(num_trajectories):
        obs, info = env.reset()
        num_generators = obs['gen_p'].shape[0]
        num_loads = obs['load_p'].shape[0]
        num_lines = obs['p_or'].shape[0]
        num_bus_objs = num_generators + num_loads + 2 * num_lines

        trajectory = {
            "timestep": [],
            "state": [],
            "action": [],
            "reward": [],
        }
        for t in range(max_timesteps):

            if random.random() < frequency:
                action = env.action_space.sample()
            else:
                action = dtutil.model_to_gym_action(np.zeros(num_bus_objs + num_lines), num_bus_objs, num_generators, num_lines) # do nothing action

            next_obs, reward, terminated, truncated, info = env.step(action)

            action_vec = dtutil.gym_to_model_action(action, num_bus_objs, num_lines)
            observation_vec = dtutil.gym_to_model_observation(obs)

            if terminated or truncated:
                break

            trajectory["timestep"].append(t)
            trajectory["state"].append(observation_vec)
            trajectory["action"].append(action_vec)
            trajectory["reward"].append(reward)

            obs = next_obs

        filename = f"./data/trajectories/random_{frequency}__{int(round((datetime.utcnow() - datetime(1970,1,1)).total_seconds()))}.csv"
        print(f"Generated Freq {frequency} Random Trajectory {r} - Lasted {t + 1} timesteps | saved to {filename}")
        df = pd.DataFrame.from_dict(trajectory)
        df.to_csv(filename, header=False, index=False)


max_timesteps = 5 * 24 * 60 // 5 # Run for 5 days
generate_full_random_trajectories(10, max_timesteps)
generate_freq_random_trajectories(10, max_timesteps, 0.1)