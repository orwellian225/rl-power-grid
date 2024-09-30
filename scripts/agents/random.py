from scripts.metrics import Metrics

class RandomAgent:

    def __init__(self, action_space):
        self.action_space = action_space
        self.metrics = Metrics(24 * 60 // 5, "Test Random Agent")

    def act(self):
        return self.action_space.sample()

    def update(self, state, action, reward):
        self.metrics.step(reward, 0.)
