import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

#from ppo_env import Gym2OpEnv  # Assuming you move the environment class to another file
from env_1 import Gym2OpEnv

def train_ppo(total_timesteps=40000, save_path="ppo_grid2op"):
    # Initialize the environment
    env = Gym2OpEnv()

    # Check the environment to make sure it's compatible with Stable Baselines3
    check_env(env, warn=True)

    # Create the PPO model
    model = PPO("MultiInputPolicy", env, verbose=1, n_epochs=10, batch_size=128, n_steps=1280)

    # Train the PPO agent
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model.save(save_path)

    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train_ppo(4000)
