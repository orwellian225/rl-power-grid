from scripts.agents.dqn import DQNAgent
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from scripts.dqn_env import Gym2OpEnv
import csv

# Custom callback to log specific metrics (train/loss and timesteps)
class LossAndTimestepCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(LossAndTimestepCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.losses = []
        self.timesteps = []

        # Open CSV file for writing
        self.csv_file = open(f"{self.log_dir}/dqn_loss_timestep_log.csv", mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Timestep", "Loss"])

    def _on_step(self):
        # Get the current timestep and loss
        timestep = self.num_timesteps
        loss = self.model.logger.name_to_value.get('train/loss', 0.0)

        # Save loss and timestep to CSV
        self.csv_writer.writerow([timestep, loss])

        return True

    def _on_training_end(self):
        # Close the CSV file after training ends
        self.csv_file.close()

# Set up the logger (optional, can keep the default or use a different directory)
log_dir = "./logs/"

# Initialize environment and DQN agent
env = Gym2OpEnv()
dqn_agent = DQNAgent(env)

# Initialize the custom callback
loss_and_timestep_callback = LossAndTimestepCallback(log_dir)

# Train the DQN Agent with the callback
total_timesteps = 10000
dqn_agent.model.learn(total_timesteps, log_interval=1, callback=loss_and_timestep_callback)

# Save the trained model
dqn_agent.save()

# Run the trained agent and log metrics
obs, _ = env.reset()
dqn_agent.metrics.reset(365 * 24 * 60 // 5, "baseline-dqn-agent")
done = False
total_reward = 0

for t in range(365 * 24 * 60 // 5):
    action = dqn_agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Get the latest loss from the model's logger
    loss = dqn_agent.model.logger.name_to_value.get('train/loss', 0.0)
    
    dqn_agent.update(obs, action, reward)
    total_reward += reward
    if terminated or truncated:
        print(t)
        break

print(f"Total reward: {total_reward}")

# Save and plot metrics
dqn_agent.metrics.save()
dqn_agent.metrics.plot(show=True)
