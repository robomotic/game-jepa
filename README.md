# Game JEPA - Joint Embedding Predictive Architecture for Reinforcement Learning

This project implements a Joint Embedding Predictive Architecture (JEPA) approach for reinforcement learning tasks. Currently, it focuses on the CartPole environment as a proof of concept.

## Project Structure

```
game-jepa/
│
├── examples/               # Example implementations
│   ├── agents.py           # Agent implementations (JEPA Agent)
│   ├── cartpole.py         # CartPole environment example
│   └── numpy_patch.py      # Patch for numpy compatibility
│
├── .venv/                  # Virtual environment
│
└── requirements.txt        # Project dependencies
```

## CartPole Example

The CartPole example demonstrates a simple implementation of a Joint Embedding Predictive Architecture (JEPA) for reinforcement learning. This example uses the CartPole-v1 environment from OpenAI Gym.

### How It Works

The JEPA approach for CartPole works as follows:

1. **State Encoding**: The agent encodes the CartPole state (position, velocity, angle, angular velocity) into a latent embedding space.

2. **Action Prediction**: The agent predicts the next state embedding based on the current state embedding and the action taken.

3. **Policy Learning**: The agent learns a policy that maximizes rewards while also improving its ability to predict future states.

### Code Structure

The implementation consists of two main files:

1. **agents.py**: Contains the `JEPAAgent` class and helper functions:
   - `JEPAAgent`: Neural network architecture with encoder, predictor, policy, and value components
   - `collect_episode`: Function to collect experience from the environment
   - `train_step`: Function to update the agent based on collected experience

2. **cartpole.py**: Contains the main training loop and environment setup:
   - `pole()`: Simple function to test the CartPole environment
   - `jepa_pole_train()`: Main training function for the JEPA agent

### Running the Example

To run the CartPole example:

```bash
python examples/cartpole.py
```

### Performance and Limitations

While the JEPA approach shows promise for reinforcement learning tasks, the current implementation has several limitations:

1. **Convergence Issues**: The loss value plateaus around 1.0, which is not optimal compared to classical Q-learning approaches that can achieve much lower loss values and better performance.

2. **Prediction vs. Policy Trade-off**: The current implementation tries to balance between learning a good state predictor and a good policy, which can lead to suboptimal performance in both tasks.

3. **Simple Architecture**: The current neural network architecture is relatively simple and may not capture the full complexity of the environment dynamics.

4. **Limited Exploration**: The agent uses a simple softmax policy for exploration, which may not be sufficient for more complex environments.

### Future Improvements

Planned improvements for the CartPole example include:

1. **Enhanced Architecture**: Implementing more sophisticated neural network architectures for both the encoder and predictor components.

2. **Better Exploration Strategies**: Incorporating more advanced exploration strategies such as intrinsic motivation or curiosity-driven exploration.

3. **Hyperparameter Tuning**: Systematically tuning hyperparameters to improve performance.

4. **Comparison with Baselines**: Directly comparing the JEPA approach with classical RL methods like DQN, A2C, and PPO on the same tasks.

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```
   # On Linux/Mac
   source venv/bin/activate
   
   # On Windows
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Implementation Details

### JEPA Agent Architecture

The `JEPAAgent` class implements a neural network with the following components:

```python
class JEPAAgent(nn.Module):
    def __init__(self, state_dim, action_dim, z_dim=64):
        # Encoder: state -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
        )
        
        # Predictor: (embedding, action) -> next embedding
        self.action_embed = nn.Embedding(action_dim, 8)
        self.predictor = nn.Sequential(
            nn.Linear(z_dim + 8, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
        )
        
        # Policy: embedding -> action logits
        self.policy = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Value: embedding -> value
        self.value = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
```

### Training Process

The training process in the CartPole example combines reinforcement learning with predictive learning:

```python
def train_step(agent, optimizer, states, actions, rewards, next_states, dones, gamma=0.99, lambda_pred=1.0):
    # Convert lists to numpy arrays for efficiency
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(np.array(actions))
    rewards = torch.FloatTensor(np.array(rewards))
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.FloatTensor(np.array(dones))
    
    # Compute embeddings
    z_t = agent.encoder(states)
    z_next = agent.encoder(next_states)
    
    # Prediction loss
    z_next_pred = agent.predict_next_z(z_t, actions)
    prediction_loss = ((z_next_pred - z_next) ** 2).mean()
    
    # RL losses (A2C)
    with torch.no_grad():
        _, next_values = agent.get_action_value(next_states)
        next_values = next_values.squeeze(-1)
        targets = rewards + gamma * next_values * (1 - dones)
        _, values = agent.get_action_value(states)
        values = values.squeeze(-1)
        advantages = targets - values
    
    action_logits, _ = agent.get_action_value(states)
    log_probs = torch.log_softmax(action_logits, dim=-1)
    log_probs_a = log_probs[range(len(actions)), actions]
    
    policy_loss = - (log_probs_a * advantages).mean()
    value_loss = ((values - targets) ** 2).mean()
    
    # Total loss
    total_loss = policy_loss + value_loss + lambda_pred * prediction_loss
    
    # Update parameters
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()
```

The training combines three types of losses:
1. **Prediction Loss**: MSE between predicted and actual next state embeddings
2. **Policy Loss**: Actor-Critic style policy gradient loss
3. **Value Loss**: MSE between predicted and target values

### Training Results

When running the CartPole example, you'll see output similar to:

```
Episode 0, Loss: 1.7315
Episode 50, Loss: 1.3207
Episode 100, Loss: 1.0361
...
Episode 950, Loss: 0.9974
```

The loss converges to around 1.0, which indicates that the agent is learning but not achieving optimal performance. Classical Q-learning approaches can achieve much better performance on the CartPole task.

## Comparison with Classical RL

The JEPA approach differs from classical reinforcement learning methods in several ways:

1. **Representation Learning**: JEPA focuses on learning meaningful state representations through prediction, while classical RL methods like Q-learning focus directly on value estimation.

2. **Predictive Component**: JEPA explicitly learns to predict future states, which may help with planning and model-based reasoning.

3. **Combined Objectives**: JEPA combines prediction and policy learning in a single framework, while many classical RL methods separate these concerns.

### Performance Comparison

In the CartPole example, the JEPA approach shows some limitations compared to classical RL methods:

1. **Convergence**: The loss plateaus around 1.0, while classical methods like DQN can solve CartPole completely (achieving the maximum possible reward consistently).

2. **Sample Efficiency**: The current JEPA implementation may require more samples to learn compared to optimized RL algorithms.

3. **Stability**: The combined loss function may lead to training instability as the agent tries to optimize multiple objectives simultaneously.

## Future Directions

Future work on the Game JEPA project will focus on:

1. **Improved Architectures**: Exploring more sophisticated neural network architectures for both encoding and prediction.

2. **Multi-Environment Support**: Extending beyond CartPole to more complex environments like Atari games.

3. **Self-Supervised Pretraining**: Implementing self-supervised pretraining phases to improve representation learning.

4. **Hybrid Approaches**: Combining JEPA with other successful RL techniques like distributional RL or hierarchical RL.

5. **Theoretical Analysis**: Better understanding the relationship between prediction accuracy and policy performance.

## Conclusion

The Game JEPA project provides a framework for exploring how predictive architectures can be applied to reinforcement learning problems. While the current CartPole implementation shows some limitations compared to classical RL approaches, the JEPA framework offers interesting possibilities for combining representation learning with policy optimization.

By focusing on learning to predict future states in a latent space, JEPA may eventually lead to agents that can better understand the dynamics of their environments and make more informed decisions.
