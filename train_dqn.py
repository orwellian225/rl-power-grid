from scripts.agents.dqn import DQNAgent

from scripts.dqn_env import Gym2OpEnv

env = Gym2OpEnv()

dqn_agent = DQNAgent(env)

# Train the DQN Agent
total_timesteps = 10000
dqn_agent.model.learn(total_timesteps)

# Save the trained model
dqn_agent.save()

# Run the trained agent and log metrics
obs, _ = env.reset()
done = False
baseline_dqn_reward = 0

while not done:
    action = dqn_agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    dqn_agent.update(obs, action, reward)
    baseline_dqn_reward += reward
    done = terminated or truncated
    
print(f"Total reward: {baseline_dqn_reward}")

dqn_agent.metrics.save()
dqn_agent.metrics.plot()