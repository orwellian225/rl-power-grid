from datetime import datetime
import pandas as pd
import numpy as np
import random

from scripts.agents.decision_transformer.env import Gym2OpEnv
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

def generate_full_random_trajectories(name, num_trajectories, max_timesteps):
    generate_freq_random_trajectories(name, num_trajectories, max_timesteps, 1.0)

def generate_freq_random_trajectories(name, num_trajectories, max_timesteps, frequency):
    print(f"Trajectories are at most {max_timesteps} timesteps in length")
    trajectory = {
        "timestep": [],
        "state": [],
        "action": [],
        "reward": [],
    }
    for r in range(num_trajectories):
        obs, info = env.reset(seed=random.randint(0, 10000))
        num_generators = obs['gen_p'].shape[0]
        num_loads = obs['load_p'].shape[0]
        num_lines = obs['p_or'].shape[0]
        num_bus_objs = num_generators + num_loads + 2 * num_lines

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

        print(f"Generated Freq {frequency} Random Trajectory {r} - Lasted {t + 1} timesteps")
    print(f"Generated {len(trajectory['timestep'])} sar tuples")
    filename = f"./data/trajectories/{name}_{frequency}.csv"
    df = pd.DataFrame.from_dict(trajectory)
    df.to_csv(filename, header=False, index=False)


max_timesteps = 5 * 24 * 60 // 5 # Run for 5 days
generate_full_random_trajectories("small_random", 10, max_timesteps)
# generate_freq_random_trajectories(200, max_timesteps, 0.1)