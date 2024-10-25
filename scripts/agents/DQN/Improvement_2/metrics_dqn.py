import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv("logs/improvement_2_dqn_metrics.csv")

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
fig.suptitle('Improvement 2 DQN Training Metrics', fontsize=16)

# Episode Reward Mean
axes[0].set_title("Improvement 2 DQN Episode Reward Mean vs Steps")
sns.lineplot(x=data['Timestep'], y=data['Reward_Mean'], ax=axes[0], color=sns.color_palette("Set2")[0])
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('Episode Reward Mean')
axes[0].grid(True)

# Episode Length Mean
axes[1].set_title("Improvement 2 DQN Episode Length Mean vs Steps")
sns.lineplot(x=data['Timestep'], y=data['Length_Mean'], ax=axes[1], color=sns.color_palette("Set2")[1])
axes[1].set_xlabel('Timestep')
axes[1].set_ylabel('Episode Length Mean')
axes[1].grid(True)

# Loss vs Steps
axes[2].set_title("Improvement 2 DQN Loss vs Steps")
sns.lineplot(x=data['Timestep'], y=data['Loss'], ax=axes[2], color=sns.color_palette("Set2")[2])
axes[2].set_xlabel('Timestep')
axes[2].set_ylabel('Loss')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('data/visualizations/improvement-2-dqn-agent.pdf')
plt.show()
