from scripts.agents.dqn import DQNAgent
from stable_baselines3.common.evaluation import evaluate_policy

from scripts.dqn_env import Gym2OpEnv

env = Gym2OpEnv()

dqn_agent = DQNAgent(env)

# Train the DQN Agent
total_timesteps = 10000
dqn_agent.model.learn(total_timesteps)

# Save the trained model
dqn_agent.save()

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(dqn_agent.model, env, n_eval_episodes=10)
print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Run the trained agent and log metrics
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = dqn_agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    dqn_agent.update(obs, action, reward)
    total_reward += reward
    
print(f"Total reward: {total_reward}")

dqn_agent.metrics.save()
dqn_agent.metrics.plot()