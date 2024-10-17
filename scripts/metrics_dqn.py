import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_reward = pd.read_csv("data/baseline-dqn-agent.csv")
data_loss = pd.read_csv("logs/dqn_loss_timestep_log.csv")

# Find the index where the reward is -0.5
index = data_reward[data_reward['Reward'] == -0.5].index[0]

# Extract steps and rewards up to the index (inclusive)
steps_up_to_minus_half = data_reward['Step'][:index + 1]
rewards_up_to_minus_half = data_reward['Reward'][:index + 1]

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))

# Plot Rewards up to -0.5
axes[0].set_title("Rewards up to -0.5")
sns.lineplot(x=steps_up_to_minus_half, y=rewards_up_to_minus_half, ax=axes[0], c=sns.color_palette("Set2")[0])
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Reward')
axes[0].grid(True)

# Plot Reward vs Steps
axes[1].set_title("Baseline DQN Rewards vs Steps")
sns.lineplot(x='Step', y='Reward', data=data_reward, ax=axes[1], c=sns.color_palette("Set2")[1])
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Reward')
axes[1].grid(True)

# Plot Loss vs Steps
axes[2].set_title("Baseline DQN Loss vs Steps")
sns.lineplot(x='Timestep', y='Loss', data=data_loss, ax=axes[2])
axes[2].set_xlabel('Timestep')
axes[2].set_ylabel('Loss')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('data/visualizations/baseline-dqn-agent.pdf')
plt.show()
