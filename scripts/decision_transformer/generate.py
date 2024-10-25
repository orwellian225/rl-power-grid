import sys
import csv
import random

import numpy as np

from agent_lookup import agents
from environment import Gym2OpEnv

if len(sys.argv) != 4:
    print("No impovement and configuration specified")
    exit()

agent_improvement = sys.argv[1]
agent_configuration = int(sys.argv[2])
num_trajectories = int(sys.argv[3])

agent_info = agents[agent_improvement][agent_configuration]

env = Gym2OpEnv(
    modifying_bus_count=agent_info["data"]["modifiable_buses"],
    modifying_line_count=agent_info["data"]["modifiable_lines"],
    curtail_bin_counts=agent_info["data"]["redispatch_bins"],
    redispatch_bin_counts=agent_info["data"]["curtail_bins"]
)

random_frequency = agent_info["data"]["random_frequency"]
masked = agent_info["data"]["masked"]
max_episode_len = agent_info["hyperparams"]["max_episode_len"]

print(f"Trajectories are at most {max_episode_len} timesteps in length")
trajectory = {
    "timestep": [],
    "state": [],
    "action": [],
    "reward": [],
}

for r in range(num_trajectories):
    obs, _ = env.reset()
    returns = 0
    for t in range(max_episode_len):

        if random.random() < random_frequency:
            action = env.action_space.sample()
        else:
            action = env.action_space.no_action()

        if masked:
            num_masked_actions = random.randint(0, env.action_space.shape[0] - 1)
            mask = np.ones(env.action_space.shape, dtype=np.int32)
            mask[np.random.randint(0, env.action_space.shape[0], num_masked_actions)] = 0
            action *= mask

        next_obs, reward, terminated, truncated, info = env.step(action)

        trajectory["timestep"].append(t)
        trajectory["state"].append(obs)
        trajectory["action"].append(action)
        trajectory["reward"].append(reward)

        obs = next_obs
        returns += reward

        if terminated or truncated:
            break

    print(f"Generated Freq {random_frequency} Random Trajectory {r} - Lasted {t + 1} timesteps with return {returns}")
print(f"Generated {len(trajectory['timestep'])} sar tuples")

with open(agent_info["meta"]["data_file"], "a+") as f:
    csvw = csv.writer(f)

    for t in range(len(trajectory["timestep"])):
        csvw.writerow([
            trajectory["timestep"][t],
            trajectory["state"][t],
            trajectory["action"][t],
            trajectory["reward"][t],
        ])
