# DQN Agent for Grid2Op Environment

## Components

### 1. Environment (scripts/dqn_env.py)

#### Observation Space
The observation space is currently using the default CompleteObservation from Grid2Op. The `setup_observations` method is a placeholder for future customization.

#### Action Space
The action space has been customized to focus on specific grid operations:

1. `act_attr_to_keep`: Specifies which types of actions the agent can take:
   - `"set_line_status_simple"`: Allows changing the status of power lines (on/off).
   - `"set_bus"`: Allows changing the bus configuration of substations.

2. `DiscreteActSpace`: Converts the complex Grid2Op action space into a discrete space compatible with gym environments.

3. `Discrete(self._gym_env.action_space.n)`: Creates a gym Discrete space with `n` possible actions.

This setup allows the agent to focus on key grid management actions while simplifying the action space for easier learning.

### 2. DQN Agent (scripts/agents/dqn.py)

- The `DQNAgent` class encapsulates the DQN model from Stable Baselines3.
- It's initialized with specific hyperparameters for the DQN algorithm.
- The agent includes methods for acting, updating, saving, and loading the model.
- We've integrated a `Metrics` class to track and visualize the agent's performance.

### 3. Training Script (train_dqn.py)

- This script ties everything together for training and evaluating the DQN agent.
- It creates the environment and agent, then trains the agent for a specified number of timesteps.
- After training, it saves the model and runs an evaluation episode.
- The evaluation results are logged and plotted using the `Metrics` class.

## Improvement 1

1. **Observation Space**: Refine the observation space to provide more relevant information to the agent.
2. **Action Space**: Further customize the action space to better suit the specific challenges of the power grid environment and avoid taking illegal actions.

Refining the observation and action spaces is crucial for enhancing the DQN agent's performance in the Grid2Op environment. By carefully selecting relevant features for the observation space, we can provide the agent with more meaningful information about the grid's state, potentially leading to faster learning and better decision-making. Similarly, customizing the action space allows us to focus on the most impactful and realistic actions in power grid operations. These improvements can result in an agent that learns more efficiently, makes more informed decisions, and better addresses the specific challenges of power grid management.
