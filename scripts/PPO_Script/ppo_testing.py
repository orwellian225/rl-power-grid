import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import os
import pandas as pd

#from ppo_env import Gym2OpEnv  # Import the custom environment
#from env_1 import Gym2OpEnv
from env_2 import Gym2OpEnv

def load_model(agent_type, model_path, env):
    if agent_type == 'PPO':
        return PPO.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

def plot_training_progress(log_dir, save_path=None):
    # Look for the monitor.csv in the log directory (this is where stable-baselines3 stores training info)
    monitor_file = os.path.join(log_dir, "monitor4.csv")
    if not os.path.exists(monitor_file):
        print(f"No monitor file found at {monitor_file}. Ensure training logs are being saved.")
        return

    # Load the monitor.csv file
    data = pd.read_csv(monitor_file, skiprows=1)  # Skip the first comment row

    # Plot the episode reward mean and episode length mean over time
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(data['l'][::15], label='Episode Length (ep_len_mean)')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Length')
    plt.title('Episode Length over Time')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(data['r'][::15], label='Episode Reward (ep_rew_mean)')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Episode Reward over Time')
    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Charts saved at: {save_path}")
    plt.close()

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
    log_dir = "/Users/konstantinoshatzipanis/Documents/Education/Wits_BSc_Degree/Honours (Mathematics)/Reinforcement Learning/Project/scripts/PPO_Script/"
    save_path = "ppo_training_progress4.png"  # Specify the path where you want to save the chart
    env = Gym2OpEnv()
    # Plot the training progress
    plot_training_progress(log_dir, save_path)

    # Example to test the PPO agent
    test_agent(env, agent_type='PPO', model_path="ppo_grid2op4", steps=1)
    env.close()

    # Example to test the random agent
    # test_random_agent(env, 1)
    # env.close()
    # env = Gym2OpEnv()
    # # Example to test the do-nothing agent
    # test_do_nothing_agent(env, 1)
    # #test_do_nothing_agent(env, 1)
    # #test_do_nothing_agent(env, 1)
    # env.close()
    # env = Gym2OpEnv()
    # # Example to test PPO agent
    # test_agent(env, agent_type='PPO', model_path="ppo_grid2op", steps=1)
    # env.close()
