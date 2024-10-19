from datetime import datetime
import numpy as np
import random
import csv

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
        obs, info = env.reset()
        num_generators = obs['gen_p'].shape[0]
        num_loads = obs['load_p'].shape[0]
        num_lines = obs['p_or'].shape[0]
        num_bus_objs = num_generators + num_loads + 2 * num_lines

        returns = 0
        for t in range(max_timesteps):

            if random.random() < frequency:
                action = env.action_space.sample()
            else:
                action = env.action_space.no_action()

            next_obs, reward, terminated, truncated, info = env.step(action)

            observation_vec = dtutil.gym_to_model_observation(obs)

            trajectory["timestep"].append(t)
            trajectory["state"].append(observation_vec)
            trajectory["action"].append(action)
            trajectory["reward"].append(reward)

            returns += reward

            if terminated or truncated:
                break

            obs = next_obs

        print(f"Generated Freq {frequency} Random Trajectory {r} - Lasted {t + 1} timesteps with return {returns}")
    print(f"Generated {len(trajectory['timestep'])} sar tuples")
    filename = f"./data/trajectories/{name}_{frequency}.csv"
    with open(filename, "a+") as f:
        csvw = csv.writer(f, lineterminator="\r")

        for t in range(len(trajectory["timestep"])):
            csvw.writerow([
                trajectory["timestep"][t],
                trajectory["state"][t],
                trajectory["action"][t],
                trajectory["reward"][t],
            ])

def generate_freq_masked_random_trajectories(name, num_trajectories, max_timesteps, max_unmasked, frequency):
    print(f"Trajectories are at most {max_timesteps} timesteps in length")
    trajectory = {
        "timestep": [],
        "state": [],
        "action": [],
        "reward": [],
    }
    for r in range(num_trajectories):
        obs, info = env.reset()
        num_generators = obs['gen_p'].shape[0]
        num_loads = obs['load_p'].shape[0]
        num_lines = obs['p_or'].shape[0]
        num_bus_objs = num_generators + num_loads + 2 * num_lines

        returns = 0
        for t in range(max_timesteps):

            if random.random() < frequency:
                action = env.action_space.sample()
            else:
                action = env.action_space.no_action()

            action_mask = np.zeros_like(action, dtype=np.int32)
            unmasked = np.random.randint(0, max_unmasked)
            for _ in range(unmasked):
                action_mask[np.random.randint(1, action_mask.shape[0])] = 1
            action = action * action_mask
            next_obs, reward, terminated, truncated, info = env.step(action)

            observation_vec = dtutil.gym_to_model_observation(obs)

            trajectory["timestep"].append(t)
            trajectory["state"].append(observation_vec)
            trajectory["action"].append(action)
            trajectory["reward"].append(reward)

            returns += reward

            if terminated or truncated:
                break

            obs = next_obs

        print(f"Generated Freq {frequency} Masked Random Trajectory {r} - Lasted {t + 1} timesteps with return {returns}")
    print(f"Generated {len(trajectory['timestep'])} sar tuples")
    filename = f"./data/trajectories/{name}_{frequency}.csv"
    with open(filename, "a+") as f:
        csvw = csv.writer(f, lineterminator="\r")

        for t in range(len(trajectory["timestep"])):
            csvw.writerow([
                trajectory["timestep"][t],
                trajectory["state"][t],
                trajectory["action"][t],
                trajectory["reward"][t],
            ])

max_timesteps = 10 * 24 * 60 // 5 # Run for 10 days
# generate_freq_masked_random_trajectories("reduced_action_masked_random", 2000, max_timesteps, 6, 0.8)
generate_freq_random_trajectories("reduced_action_random", 2000, max_timesteps, 1.0)