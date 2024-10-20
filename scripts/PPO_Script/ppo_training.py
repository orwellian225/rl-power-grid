import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import os

#from ppo_env import Gym2OpEnv  # Assuming you move the environment class to another file
#from env_1 import Gym2OpEnv
from env_2 import Gym2OpEnv

def train_ppo(total_timesteps=40000, save_path="ppo_grid2op4", load_existing=False):
    # Initialize the environment and wrap it in Monitor to log statistics
    env = Monitor(Gym2OpEnv(), filename="/Users/konstantinoshatzipanis/Documents/Education/Wits_BSc_Degree/Honours (Mathematics)/Reinforcement Learning/Project/scripts/PPO_Script/monitor4")

    # Check the environment to make sure it's compatible with Stable Baselines3
    check_env(env, warn=True)

    # Load the existing model if specified, otherwise create a new one
    if load_existing and os.path.exists(f"{save_path}.zip"):
        print(f"Loading existing model from {save_path}")
        model = PPO.load(save_path, env=env)  # Load the saved model and pass the environment
    else:
        print("Creating a new model")
        model = PPO("MultiInputPolicy", env, verbose=1, n_epochs=10, batch_size=128, n_steps=1280)

    # Continue training the PPO agent
    model.learn(total_timesteps=total_timesteps)

    # Save the updated model
    model.save(save_path)

    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    # Set `load_existing=True` if you want to continue training the previously saved model
    train_ppo(50000,"ppo_grid2op4",load_existing=False)
