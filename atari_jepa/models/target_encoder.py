"""
Target Encoder for JEPA model.
Encodes actions into a latent representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetEncoder(nn.Module):
    """
    Target Encoder for the JEPA model.
    Encodes actions into a latent representation.
    """
    
    def __init__(self, action_dim=18, embedding_dim=256):
        """
        Initialize the target encoder.
        
        Args:
            action_dim (int): Dimension of the action space (18 for Atari)
            embedding_dim (int): Dimension of the output embeddings
        """
        super(TargetEncoder, self).__init__()
        
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        
        # MLP for encoding actions
        self.fc1 = nn.Linear(action_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, embedding_dim)
        
        # Normalization layer
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        """
        Forward pass of the target encoder.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, action_dim)
            
        Returns:
            Tensor: Encoded embeddings of shape (batch_size, embedding_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Normalize embeddings
        x = self.norm(x)
        
        return x


class AtariActionEncoder(nn.Module):
    """
    Alternative implementation of the Target Encoder.
    Uses a simpler architecture for encoding discrete actions.
    """
    
    def __init__(self, action_dim=18, embedding_dim=256):
        """
        Initialize the Atari action encoder.
        
        Args:
            action_dim (int): Number of discrete actions in Atari (18)
            embedding_dim (int): Dimension of the output embeddings
        """
        super(AtariActionEncoder, self).__init__()
        
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        
        # Embedding layer for discrete actions
        self.embedding = nn.Embedding(action_dim, embedding_dim)
        
        # Projection layer to match embedding dimensions
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Normalization layer
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x_idx):
        """
        Forward pass of the action encoder.
        
        Args:
            x_idx (Tensor): Input tensor of shape (batch_size,) containing action indices
            
        Returns:
            Tensor: Encoded embeddings of shape (batch_size, embedding_dim)
        """
        # Get embeddings for the action indices
        x = self.embedding(x_idx)
        
        # Project to final embedding space
        x = self.projection(x)
        
        # Normalize embeddings
        x = self.norm(x)
        
        return x
    
    def forward_one_hot(self, x_one_hot):
        """
        Alternative forward pass using one-hot encoded actions.
        
        Args:
            x_one_hot (Tensor): Input tensor of shape (batch_size, action_dim) with one-hot encoding
            
        Returns:
            Tensor: Encoded embeddings of shape (batch_size, embedding_dim)
        """
        # Convert one-hot to indices
        x_idx = torch.argmax(x_one_hot, dim=1)
        
        # Use the standard forward pass
        return self.forward(x_idx)
