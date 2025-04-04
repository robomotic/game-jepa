import torch
import torch.nn as nn

import gym
import numpy as np

class JEPAAgent(nn.Module):
    """
    NOTE: This is not possibly the correct implementation!!!!

    """
    def __init__(self, state_dim, action_dim, z_dim=64):
        super(JEPAAgent, self).__init__()

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

    def get_action_value(self, state):
        """Get action logits and value from state for action selection."""
        z = self.encoder(state)
        action_logits = self.policy(z)
        value = self.value(z)
        return action_logits, value

    def predict_next_z(self, z, action):
        """Predict next embedding from current embedding and action."""
        a_emb = self.action_embed(action)
        input_pred = torch.cat([z, a_emb], dim=-1)
        z_next_pred = self.predictor(input_pred)
        return z_next_pred

def collect_episode(agent, env):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    state, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_logits, _ = agent.get_action_value(state_tensor)
            action_prob = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_prob, 1).item()

        next_state, reward, terminated, truncated, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(terminated or truncated)

        state = next_state

    return states, actions, rewards, next_states, dones


def train_step(agent, optimizer, states, actions, rewards, next_states, dones, gamma=0.99, lambda_pred=1.0):
    # Convert lists to numpy arrays first to avoid the slow tensor creation warning
    import numpy as np
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