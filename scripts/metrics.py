import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv

class Metrics:
    def __init__(self, num_steps: int, run_name: str):
        self.reset(num_steps, run_name)

    def step(self, reward: float, loss: float):
        self.step_rewards[self.current_step] = reward
        self.step_losses[self.current_step] = loss
        self.current_step += 1

    def reset(self, num_steps: int, run_name: str) -> bool:
        self.run_name = run_name
        self.num_steps = num_steps
        self.current_step = 0
        
        self.step_rewards = np.zeros(num_steps)
        self.step_losses = np.zeros(num_steps) 

    def save(self):
        file = open(f"./data/{self.run_name.lower().replace(' ', '-')}.csv", "w+", newline="")
        csvw = csv.writer(file)

        csvw.writerow(["Step", "Reward", "Loss"])
        for step_i in range(self.num_steps):
            csvw.writerow([step_i, self.step_rewards[step_i], self.step_losses[step_i]])

    def plot(self, show: bool = True, save: bool = True):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

        plt.suptitle(f"{self.run_name} Performance")

        steps = np.arange(0, self.num_steps)

        axes[0].set_title("Reward at step t")
        axes[0].plot(steps, self.step_rewards, c=sns.color_palette("Set2")[0])
        axes[0].set_ylabel("Reward")
        axes[0].set_xlabel("Number of Steps")

        axes[1].set_title("Loss at step t")
        axes[1].plot(steps, self.step_losses, c=sns.color_palette("Set2")[1])
        axes[1].set_ylabel("Loss")
        axes[1].set_xlabel("Number of Steps")

        if show:
            plt.show()

        if save:
            fig.savefig(f"./data/visualizations/{self.run_name.lower().replace(' ', '-')}.pdf")

        plt.close()