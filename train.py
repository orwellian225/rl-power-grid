from scripts.agents.random import RandomAgent

from scripts.env import Gym2OpEnv
env = Gym2OpEnv()

random_agent = RandomAgent(env.action_space)

state, info = env.reset()
for _ in range(24 * 60 // 5): # Run for one day
    action = random_agent.act()
    obs, reward, terminated, truncated, info = env.step(action)
    random_agent.update(obs, action, reward)

random_agent.metrics.save()
random_agent.metrics.plot()