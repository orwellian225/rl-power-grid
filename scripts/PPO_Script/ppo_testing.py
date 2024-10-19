import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from ppo_env import Gym2OpEnv  # Import the custom environment
from env_1 import Gym2OpEnv  # Import the custom environment


def load_model(agent_type, model_path, env):
    if agent_type == 'PPO':
        return PPO.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

def test_agent(env, agent_type='PPO', model_path="ppo_grid2op", steps=2000):
    # Load the saved model
    model = load_model(agent_type, model_path, env)

    # Test the trained model
    obs, _ = env.reset(seed=1)
    total_reward = 0
    for step in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step}: PPO agent took action: {action}, received reward: {reward}")
        if terminated or truncated:
            #obs, _ = env.reset(seed=2)
            break

    print(f"{agent_type} agent reward: {total_reward}")
    print(f"Test steps: {steps}")


def test_random_agent(env, steps=2000):
    # Test the random agent
    obs, _ = env.reset(seed=1)
    total_reward = 0
    for step in range(steps):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step}: RAN agent took action: {action}, received reward: {reward}")
        if terminated or truncated:
            #obs, _ = env.reset(seed=2)
            break

    print(f"Random agent reward: {total_reward}")
    print(f"Test steps: {steps}")


def test_do_nothing_agent(env, steps=2000):
    # Assuming the 'do nothing' action is an array of zeros (adjust based on your action space)
    no_action = np.zeros_like(env.action_space.nvec)

    # Test the do-nothing agent
    obs, _ = env.reset(seed=1)
    total_reward = 0
    for step in range(steps):
        obs, reward, terminated, truncated, info = env.step(no_action)
        total_reward += reward
        print(f"Step {step}: DN agent took action: {no_action}, received reward: {reward}")
        if terminated or truncated:
            #obs, _ = env.reset(seed=2)
            break

    print(f"Do-Nothing agent reward: {total_reward}")
    print(f"Test steps: {steps}")


if __name__ == "__main__":
    env = Gym2OpEnv()

    # Example to test the random agent
    test_random_agent(env, 1)
    env.close()
    env = Gym2OpEnv()
    # Example to test the do-nothing agent
    test_do_nothing_agent(env, 1)
    #test_do_nothing_agent(env, 1)
    #test_do_nothing_agent(env, 1)
    env.close()
    env = Gym2OpEnv()
    # Example to test PPO agent
    test_agent(env, agent_type='PPO', model_path="ppo_grid2op", steps=1)
    env.close()
