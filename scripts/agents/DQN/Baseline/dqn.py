from stable_baselines3 import DQN
from scripts.metrics import Metrics

class DQNAgent:

    def __init__(self, env):
        # Initialise DQN agent
        self.env = env
        self.model = DQN(
            "MultiInputPolicy", 
            env,
            learning_rate=1e-3,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=32,
            tau=0.1,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            verbose=1
        )

        # Initialise metrics for tracking performance
        self.metrics = Metrics(365 * 24 * 60 // 5, "Baseline DQN Agent")

    def act(self, obersevation):
        # Predict the action from the given obersevation
        action, _states = self.model.predict(obersevation)
        return action
    
    def update(self, obs, action, reward):
        # Update the model with the given obs, action and reward
        self.metrics.step(reward, 0.)

    def save(self, filename="scripts/agents/DQN/Baseline/baseline_dqn_agent"):
        # Save the trained model
        self.model.save(filename)

    def load(self, filename="scripts/agents/DQN/Baseline/baseline_dqn_agent"):
        # Load the trained model
        self.model = DQN.load(filename)