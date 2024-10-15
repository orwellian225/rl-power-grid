from scripts.agents.dqn import DQNAgent
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from scripts.dqn_env import Gym2OpEnv

# Set up the logger to log to a CSV file
log_dir = "./logs/"
logger = configure(log_dir, ["stdout", "csv"])

# Initialize environment and DQN agent
env = Gym2OpEnv()
dqn_agent = DQNAgent(env)

# Set the logger for the model
dqn_agent.model.set_logger(logger)

# Train the DQN Agent
total_timesteps = 10000
dqn_agent.model.learn(total_timesteps, log_interval=1)

# Save the trained model
dqn_agent.save()

# Run the trained agent and log metrics
obs, _ = env.reset()
dqn_agent.metrics.reset(365 * 24 * 60 // 5, "Evaluation")
done = False
baseline_dqn_reward = 0

for t in range(365 * 24 * 60 // 5):
    action = dqn_agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Get the latest loss from the CSV file
    # This will be handled automatically by the logger output
    dqn_agent.update(obs, action, reward)
    total_reward += reward
    if terminated or truncated:
        print(t)
        break

print(f"Total reward: {total_reward}")

# Save and plot metrics
dqn_agent.metrics.save()
dqn_agent.metrics.plot(show=True)