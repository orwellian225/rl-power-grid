from sb3_contrib import QRDQN
from scripts.metrics import Metrics

class DQNAgent:

    def __init__(self, env):
        # Initialise QRDQN agent
        self.env = env
        self.model = QRDQN(
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
        self.metrics = Metrics(24 * 60 // 5, "Improvement 2 DQN Agent")

    def act(self, observation):
        # Predict the action from the given observation
        action, _states = self.model.predict(observation)
        return action
    
    def update(self, obs, action, reward):
        # Update the model with the given obs, action and reward
        self.metrics.step(reward, 0.)

    def save(self, filename="scripts/agents/DQN/Improvement_2/improvement_2_dqn_agent"):
        # Save the trained model
        self.model.save(filename)

    def load(self, filename="scripts/agents/DQN/Improvement_2/improvement_2_dqn_agent"):
        # Load the trained model
        self.model = QRDQN.load(filename)
