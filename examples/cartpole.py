# Import numpy patch first to fix the bool8 issue
import numpy_patch

import gym
from agents import JEPAAgent,collect_episode,train_step
import torch.optim as optim

def pole():
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()

def jepa_pole_train():
    # Set up environment and agent
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    assert state_dim == 4
    action_dim = env.action_space.n
    assert action_dim==2
    # starting from a latent dimenion of 64 .... but how do we pick this?
    z_dim = 64

    agent = JEPAAgent(state_dim, action_dim, z_dim)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)

    # Training loop
    num_episodes = 1000
    for episode in range(num_episodes):
        states, actions, rewards, next_states, dones = collect_episode(agent, env)
        loss = train_step(agent, optimizer, states, actions, rewards, next_states, dones)

        if episode % 50 == 0:
            print(f"Episode {episode}, Loss: {loss:.4f}")
    env.close()

# If run directly
if __name__ == "__main__":
    jepa_pole_train()