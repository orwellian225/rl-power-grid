import csv
import numpy as np

def load_data(filname, _print=False):
    file = open(filname, "r")
    csvr = csv.reader(file)

    num_sar_tuples = 0

    timesteps = []
    states = []
    actions = []
    rewards = []

    for line in csvr:
        num_sar_tuples += 1

        timestep = int(line[0])
        observation = np.array(line[1][2:-1].split(), dtype=np.float32)
        action = np.array(line[2][1:-1].split(), dtype=np.float32)
        reward = float(line[3])

        timesteps.append(timestep)
        states.append(observation)
        actions.append(action)
        rewards.append(reward)

    if _print:
        print(f"Loaded {num_sar_tuples} sar tuples")
    timesteps = np.array(timesteps)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    if _print:
        print(f"State Dim: {states.shape}")
        print(f"Action Dim: {actions.shape}")
        print(f"Reward Dim: {rewards.shape}")

    return timesteps, states, actions, rewards