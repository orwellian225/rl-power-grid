from grid2op.Agent import DoNothingAgent, RandomAgent
from scripts.env import Gym2OpEnv

env = Gym2OpEnv()

random_agent = RandomAgent(env.action_space)
do_nothing_agent = DoNothingAgent(env.action_space)

done = False
time_step = int(0)
cum_reward = 0.
obs = env.reset()
reward = env.reward_range[0]
max_iter = 10

while not done:
    act = do_nothing_agent.act(obs, reward, done) # chose an action to do, in this case "do nothing"
    obs, reward, done, info = env.step(act) # implement this action on the powergrid
    cum_reward += reward
    time_step += 1
    if time_step >= max_iter:
        break

print("This agent managed to survive {} timesteps".format(time_step))
print("It's final cumulated reward is {}".format(cum_reward))




# state, info = env.reset()
#
# for _ in range(24 * 60 // 5): # Run for one day
#     action = random_agent.act()
#     obs, reward, terminated, truncated, info = env.step(action)
#     random_agent.update(obs, action, reward)
#
# random_agent.metrics.save()
# random_agent.metrics.plot()