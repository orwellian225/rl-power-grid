# Implementation Details for DQN Agents in Grid2OP Environment

## Baseline DQN Implementation

The baseline DQN algorithm is implemented using the `stable-baselines3` library, which provides a reliable baseline to use for this reinforcement learning task. The agent is trained to interact with the environment and optimise grid operations through learning from rewards.

### DQN Agent Implementation

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
- The **target network** updates, reduce the likelihood of training instability, common in reinforcement learning tasks with high-dimensional state spaces.

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

The DQN algorithm, paired with the `MultiInputPolicy`, allows for handling the complex and multi-dimensional observation space. The discrete action space simplifies exploration, ensuring that the agent focuses on meaningful grid control operations.


## Improvement 1: Action and Observation Space Pruning

### Observation Space Pruning
The original observation space from Grid2Op includes a wide variety of attributes, many of which may not be directly relevant or useful for decision-making by the agent. By default, this can result in unnecessary complexity and increase the dimensionality of the input, which can slow down training and affect the agent's ability to generalise. 

#### **Improvements:**
The observation space has been pruned to focus on key attributes that directly influence decision-making:

- **Retained Attributes:**
   - `rho`: Power flow divided by thermal limits for each line, which is crucial to prevent overloading.
   - `gen_p`: Active power production, normlised by the maximum generator capacity. This gives the agent an understanding of generation constraints.
   - `load_p`: Active power consumption, normalised relative to the initial load, helping the agent prioritise load-shedding decisions.
   - `topo_vect`: The current topology vector, showing how substations are interconnected, crucial for topology reconfiguration.
- **Removed Attributes:** All other attributes deemed less relevant for the core decision-making process were excluded, significantly reducing the dimensionality of the observation space.

- **Custom Observations:**
   - **Connectivity Matrix**: Represents the network's structure, helping the agent make better decisions about power flow management.
   - **Line Capacity Usage**: The percentage of line capacity being used, giving clear indication of how close each line it to its limits.
   - **Line Overflow**: A binary indicator that signals if any lines are overloaded, allowing the agent to take immediate corrective action.

#### **Rationale:**
Pruning the observation space focuses the agent's attention on the most critical aspects of the environment, which can lead to:
   - **Faster Learning**: With fewer input dimensions, the agent can process states more efficiently, leading to faster convergence.
   - **Improved Generalisation**: By focusing on key attributes, the agent is less likely to overfit to irrelevant parts of the observation space, making it more robust in different scenarios.

### Action Space Pruning and Discretization

The action space of the original Grid2Op environment includes a large number of possible actions, including setting bus configurations, changing line statuses, redispatching power, and curtailing load. This results in an extremely large and complex action space, especially when continuous actions are considered.

#### **Improvements:**
The action space has been pruned and discretized to make it more manageable:
- **Removed Actions**:
  - **`set_bus`**: This action, which sets the bus configuration of substations, was removed. While important, managing bus configurations introduces a large number of possibilities that can complicate the agent's decision-making process.
  - **`set_line_status`**: The ability to change line statuses was also removed to simplify the action space. Instead, the agent focuses on managing generation and load.
  
- **Discretization of Actions**:
  - **`curtail`**: Continuous curtailment values were discretized into three bins. This reduces the complexity of the curtailment action, enabling the agent to make coarse adjustments instead of precise, continuous ones.
  - **`redispatch`**: Similarly, redispatching power was discretized into five bins, allowing the agent to adjust power generation in a step-wise manner rather than selecting an exact value from a continuous range.

- **Discrete Action Space**:
  After pruning and discretization, the remaining actions were converted into a discrete action space using `DiscreteActSpace`, making it easier for the agent to explore and select actions during training.

#### **Rationale:**
Pruning and discretizing the action space was a crucial improvement for the following reasons:
- **Reduced complexity**: By removing certain actions and discretizing others, the size of the action space has been dramatically reduced. This allows the agent to focus on a smaller set of critical actions, making the learning process more efficient.
- **Faster learning**: With a simpler and smaller action space, the agent can explore actions more effectively and avoid getting stuck in exploration bottlenecks, leading to faster learning and convergence.
- **Improved robustness**: Simplifying the action space helps reduce the likelihood of the agent making invalid or ineffective actions. For example, discretizing curtailment and redispatch ensures that the agent only attempts feasible adjustments, while removing bus and line status changes reduces the chance of destabilizing the network.

### **Why These Changes Are a Good Idea**

1. **Focusing on Core Tasks**:
   The environment has been simplified by focusing the observation and action spaces on the most critical aspects of grid management: managing load, generation, and topology. This reduction in complexity allows the agent to focus on core tasks without being overwhelmed by unnecessary details.

2. **Efficiency in Training**:
   By reducing the dimensionality of the observation space and simplifying the action space, the agent can learn more quickly and efficiently. Fewer irrelevant inputs and a smaller, more manageable action space means the agent can explore the environment more effectively, resulting in faster convergence.

3. **Improved Decision-Making**:
   The agent is now better equipped to make decisions related to line overloads, generation redispatch, and load curtailment. The added custom observations, like the connectivity matrix and line overflows, give the agent deeper insights into the current grid state, which can improve its ability to prevent cascading failures or optimize power distribution.

4. **Scalability**:
   These changes also make the environment more scalable. A large, unpruned action and observation space can cause performance bottlenecks, especially as the grid size grows. The current modifications ensure the environment remains scalable and efficient, even in more complex grid scenarios.


## Improvement 2: Transition to QRDQN Architecture

### Overview
In Improvement 2, we transitioned from a standard DQN implementation to a QRDQN (Quantile Regression DQN) architecture. This change enhances the agent's performance by allowing it to better capture the distribution of returns, rather than just estimating the expected value.

### Key Architectural Changes

1. **Model Type**:
   - **DQN**: Estimates the expected value of the action-value function with a single output for each action.
   - **QRDQN**: Predicts multiple quantiles for each action, providing a richer representation of the return distribution.

2. **Output Layer**:
   - **DQN Output**: A single Q-value for each action.
   - **QRDQN Output**: Multiple quantiles for each action, giving more detailed insights into possible returns.

3. **Loss Function**:
   - **DQN**: Uses mean squared error (MSE) to minimize the difference between predicted and target Q-values.
   - **QRDQN**: Uses a quantile loss function, which is more robust to outliers and helps stabilize training.

4. **Exploration**:
   - **DQN**: Relies on epsilon-greedy strategies, which can lead to suboptimal exploration.
   - **QRDQN**: Leverages the distribution of returns to inform more effective exploration strategies.

### Implications of the Architectural Change

- **Improved Performance**: QRDQN’s ability to model the return distribution allows for more informed decision-making, leading to improved learning speed and policy quality.
- **Better Handling of Uncertainty**: The architecture better captures the variability in outcomes, making it suitable for environments with high uncertainty.
- **Enhanced Robustness**: The quantile regression approach makes the agent more robust to outliers and noise in the reward signals.

### Rationale for Changes

1. **Capturing Uncertainty**: QRDQN models multiple quantiles, enabling the agent to better handle uncertainty, which is crucial in complex environments like Grid2Op.
   
2. **Improved Decision-Making**: The richer output representation helps the agent make more nuanced decisions based on the distribution of returns rather than relying on expected values alone.

3. **Stability in Training**: The quantile loss function reduces the impact of outliers and provides a more reliable learning signal, improving training stability.

4. **Enhanced Exploration Strategies**: By modeling the distribution of returns, QRDQN facilitates more informed exploration strategies, allowing the agent to balance high-risk and low-risk actions better.