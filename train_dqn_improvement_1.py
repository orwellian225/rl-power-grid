from scripts.agents.DQN.Improvement_1.dqn import DQNAgent
from stable_baselines3.common.callbacks import BaseCallback
from scripts.agents.DQN.Improvement_1.dqn_env import Gym2OpEnv
import csv
import numpy as np
from datetime import datetime
import os

class SimpleMetricsCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(SimpleMetricsCallback, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create metrics file
        self.metrics_file = open(f"{self.log_dir}/improvement_1_dqn_metrics.csv", mode='w', newline='')
        self.writer = csv.writer(self.metrics_file)
        self.writer.writerow([
            "Timestamp",
            "Timestep",
            "Loss",
            "Reward_Mean",
            "Length_Mean"
        ])
        
        # Initialize tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Update episode tracking
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Get loss from logger
        loss = self.model.logger.name_to_value.get('train/loss', 0.0)
        
        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Calculate running means
        reward_mean = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        length_mean = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        
        # Log metrics
        self.writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.num_timesteps,
            loss,
            reward_mean,
            length_mean
        ])
        return True

    def _on_training_end(self):
        self.metrics_file.close()

log_dir = "./logs/"

env = Gym2OpEnv()
dqn_agent = DQNAgent(env)

# Initialize the custom callback
metrics_callback = SimpleMetricsCallback(log_dir)

# Train the DQN Agent with the callback
total_timesteps = 50000
dqn_agent.model.learn(total_timesteps, log_interval=5, callback=metrics_callback)

# Save the trained model
dqn_agent.save()

# Evaluate the trained model

# Define number of runs for evaluation
num_runs = 5
total_steps_per_run = []
total_rewards_per_run = []

# Loop to run multiple episodes and track metrics
for run in range(num_runs):
    obs, _ = env.reset()
    dqn_agent.metrics.reset(24 * 60 // 5, f"improvement-1-dqn-agent-run-{run+1}")
    done = False
    total_reward = 0
    timesteps = 0

    for t in range(365 * 24 * 60 // 5):
        action = dqn_agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        dqn_agent.update(obs, action, reward)
        total_reward += reward
        timesteps += 1
        
        if terminated or truncated:
            print(f"Run {run+1} terminated after {timesteps} timesteps")
            break

    total_steps_per_run.append(timesteps)
    total_rewards_per_run.append(total_reward)

# Calculate average timesteps and rewards across runs
average_steps = np.mean(total_steps_per_run)
average_rewards = np.mean(total_rewards_per_run)

print(f"Average Steps over {num_runs} runs: {average_steps}")
print(f"Average total reward over {num_runs} runs: {average_rewards}")

# Save the averaged results to a CSV file
csv_filename = f"{log_dir}/improvement_1_dqn_evaluation_results.csv"
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write header
    csv_writer.writerow(["Run", "Steps", "Total Reward"])
    
    # Write results for each run
    for run in range(num_runs):
        csv_writer.writerow([f"Run {run+1}", total_steps_per_run[run], total_rewards_per_run[run]])
    
    # Write the average results
    csv_writer.writerow(["Average", average_steps, average_rewards])

print(f"Results saved to {csv_filename}")

