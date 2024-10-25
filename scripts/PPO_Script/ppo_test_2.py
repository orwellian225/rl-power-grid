import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import os
import pandas as pd

#from ppo_env import Gym2OpEnv
from env_2 import Gym2OpEnv


# Apply a moving average to smooth the data
def smooth_data(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

#Load model into the file
def load_model(agent_type, model_path, env):
    if agent_type == 'PPO':
        return PPO.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

#Plot the training data
def plot_training_progress(log_dir, save_path=None, smooth_window=10):
    monitor_file = os.path.join(log_dir, "monitor_improv2.csv")
    if not os.path.exists(monitor_file):
        print(f"No monitor file found at {monitor_file}. Ensure training logs are being saved.")
        return

    # Load the monitor.csv file
    data = pd.read_csv(monitor_file, skiprows=1)  # Skip the first comment row

    # Apply smoothing
    smoothed_lengths = smooth_data(data['l'], smooth_window)
    smoothed_rewards = smooth_data(data['r'], smooth_window)

    # Plot the smoothed data
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(smoothed_lengths, label='Episode Length (smoothed)')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Length')
    plt.title('Episode Length over Time')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(smoothed_rewards, label='Episode Reward (smoothed)')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Episode Reward over Time')
    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Charts saved at: {save_path}")
    plt.close()

# Function to plot the rewards and episode lengths of test data
def plot_test_results(all_rewards, episode_lengths, steps=2000, num_runs=10, save_path=None):
    # Extract the final cumulative reward at the end of each run (last value of each array in all_rewards)
    final_rewards = [rewards[-1] for rewards in all_rewards]

    # Create subplots: 1 row, 2 columns (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot final cumulative rewards for each run as a bar chart on the first subplot (ax1)
    ax1.bar(range(num_runs), final_rewards, color='blue')
    ax1.set_xlabel('Run Number')
    ax1.set_ylabel('Final Cumulative Reward')
    ax1.set_title(f'Final Cumulative Reward Per Run ({num_runs} runs)')
    ax1.set_xticks(range(num_runs))
    ax1.set_xticklabels([f'Run {i + 1}' for i in range(num_runs)])

    # Plot episode lengths for each run on the second subplot (ax2)
    ax2.plot(range(len(episode_lengths)), episode_lengths, label='Episode Lengths', color='green')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Episode Length')
    ax2.set_title(f'Episode Lengths Over Time ({num_runs} runs)')
    ax2.legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")

    # Close the plot to free memory
    plt.close()

# Function to test the agent
def test_agent(env, agent_type='PPO', model_path="ppo_grid2op_improv2", steps=2000, num_runs=20):
    model = load_model(agent_type, model_path, env)

    all_rewards = []  # Store cumulative rewards for each run
    episode_lengths = []  # Store episode lengths for each run

    for run in range(num_runs):
        obs, _ = env.reset(seed=run)
        total_reward = 0
        run_rewards = []  # Rewards for this specific run
        current_episode_length = 0  # Track episode length

        for step in range(steps):
            action, _states = model.predict(obs, deterministic=True)
            #print(action)
            obs, reward, terminated, truncated, info = env.step(action)
            print(info)
            total_reward += reward
            run_rewards.append(total_reward)
            current_episode_length += 1

            if terminated or truncated:
                episode_lengths.append(current_episode_length)  # Track episode length
                current_episode_length = 0  # Reset for next episode
                obs, _ = env.reset(seed=run + step)  # Reset environment for the next episode

        all_rewards.append(run_rewards)
        print(f"Run {run + 1}/{num_runs}: {agent_type} agent total reward: {total_reward}")

    # Return the results (rewards per run and episode lengths)
    return all_rewards, episode_lengths

if __name__ == "__main__":
    log_dir = "./scripts/PPO_Script/"
    save_path_train = "ppo_training_progress_improv2.png"  # Specify the path where you want to save the chart
    save_path_test = "ppo_testing_progress_improv2.png"
    env = Gym2OpEnv()
    # Plot the training progress
    plot_training_progress(log_dir, save_path_train)

    #test the PPO agent
    all_rewards, episode_lengths = test_agent(env, agent_type='PPO', model_path="ppo_grid2op_improv2.zip", steps=1440, num_runs=10)
    plot_test_results(all_rewards, episode_lengths, steps=1440, num_runs=10, save_path=save_path_test)
    env.close()
