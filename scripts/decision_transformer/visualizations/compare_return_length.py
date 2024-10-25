import sys
import csv

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if len(sys.argv) != 4:
    print("Incorrect args: agent, wandb return file & wandb length file must be specified")
    exit()

agent = sys.argv[1]
return_file = sys.argv[2]
length_file = sys.argv[3]

steps = []
returns = []
lengths = []

with open(return_file, "r") as f:
    csvr = csv.reader(f)

    for i, line in enumerate(csvr):
        if i == 0:
            continue
        
        steps.append(i)
        returns.append(float(line[1]))

with open(length_file, "r") as f:
    csvr = csv.reader(f)

    for i, line in enumerate(csvr):
        if i == 0:
            continue

        lengths.append(float(line[1]))

steps = np.array(steps)
returns = np.array(returns)
lengths = np.array(lengths)
ratio = returns / lengths

sns.set_style("whitegrid")
plt.plot(steps, ratio)
plt.title(f"{agent} Average reward per step")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.savefig(f"./data/visualizations/{agent}_reward_per_step.pdf")