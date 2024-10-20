# Implementation Details for DQN Agents in Grid2OP Environment

## Baseline DQN Implementation

The baseline DQN algorithm is implemenrted using the `stable-baselines3` library, which provides a reliable baseline to use got this reinforcement learning task. The agent is trained to interact with the environment and optimise grid operations through learning from rewards.

### DQN AgentImplementation

The `DQNAgent` class is responsible for managing the DQN model. Below, we outline the key components of the model and their respective purposes.

### Hyperparameters

- **Policy**: `"MultiInputPolicy"`
  - Chosen to handle the complex, multi-dimensional input space from the Grid2Op environment.
  
- **Learning Rate**: `1e-3`
  - A moderate learning rate to ensure stable convergence during training.

- **Buffer Size**: `10000`
  - The replay buffer stores past experiences to break temporal correlations and improve sample efficiency.

- **Learning Starts**: `1000`
  - The agent starts learning after collecting 1000 steps of data, ensuring enough experience is gathered before updates.

- **Batch Size**: `32`
  - The size of the mini-batches used for training. A smaller batch size was chosen to stabilize gradient updates.

- **Tau**: `0.1`
  - This parameter controls the soft update rate of the target network, used to stabilize training by slowly updating the target model.

- **Gamma**: `0.99`
  - The discount factor controls the agent's emphasis on future rewards, encouraging long-term planning.

- **Train Frequency**: `4`
  - The model is trained every 4 steps, ensuring that updates happen frequently without overwhelming the agent with too much new data.

- **Target Update Interval**: `1000`
  - The target network is updated every 1000 steps, which stabilizes the learning process by decoupling the target used for training from the current model being trained.

These hyperparameters were chosen as they are standard for baseline DQN implementations. They provide a balance between stability and efficiency, allowing the agent to learn effectively from a large state-action space.

- The **learning rate** is low enough to ensure convergence while still allowing the agent to learn from each step.
- The **buffer size** ensures the agent has sufficient experience to sample from, while not overloading memory.
- The **target network** updates reduce the likelihood of training instability, common in reinforcement learning tasks with high-dimensional state spaces.

### Environment Setup

#### Observation Space
The observation space is currently using the default CompleteObservation from Grid2Op. The `setup_observations` method is a placeholder for future customization.

#### Action Space
The action space has been customized to focus on specific grid operations:

1. `act_attr_to_keep`: Specifies which types of actions the agent can take:
   - `"set_line_status_simple"`: Allows changing the status of power lines (on/off).
   - `"set_bus"`: Allows changing the bus configuration of substations.

2. `DiscreteActSpace`: Converts the complex Grid2Op action space into a discrete space compatible with stable-baselines3 DQN algorithm.

3. `Discrete(self._gym_env.action_space.n)`: Creates a gym Discrete space with `n` possible actions.

This setup allows the agent to focus on key grid management actions while simplifying the action space for easier learning.

#### Rationale for Environment Design

The DQN algorithm, paired with the `MultiInputPolicy`, allows for handling the complex and multi-dimensional observation space. The discrete action space simplifies exploration, ensuring that the agent focuses on meaningful grid control operations






