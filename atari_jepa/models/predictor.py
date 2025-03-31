"""
Predictor module for JEPA model.
Predicts target embeddings from context embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class Predictor(nn.Module):
    """
    Predictor network for the JEPA model.
    Predicts action embeddings from frame embeddings.
    """
    
    def __init__(self, embedding_dim=256, hidden_dim=512):
        """
        Initialize the predictor network.
        
        Args:
            embedding_dim (int): Dimension of the input and output embeddings
            hidden_dim (int): Dimension of the hidden layer
        """
        super(Predictor, self).__init__()
        
        # MLP for predicting embeddings
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embedding_dim)
        
        # BatchNorm layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Forward pass of the predictor network.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Tensor: Predicted embeddings of shape (batch_size, embedding_dim)
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ResidualPredictor(nn.Module):
    """
    Residual predictor network for the JEPA model.
    Uses residual connections for better gradient flow.
    """
    
    def __init__(self, embedding_dim=256, hidden_dim=512, num_blocks=3):
        """
        Initialize the residual predictor network.
        
        Args:
            embedding_dim (int): Dimension of the input and output embeddings
            hidden_dim (int): Dimension of the hidden layer
            num_blocks (int): Number of residual blocks
        """
        super(ResidualPredictor, self).__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, x):
        """
        Forward pass of the residual predictor network.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Tensor: Predicted embeddings of shape (batch_size, embedding_dim)
        """
        # Initial projection
        x = F.relu(self.input_proj(x))
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for the predictor network.
    """
    
    def __init__(self, hidden_dim):
        """
        Initialize the residual block.
        
        Args:
            hidden_dim (int): Dimension of the hidden layer
        """
        super(ResidualBlock, self).__init__()
        
        # MLP layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # BatchNorm layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Forward pass of the residual block.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, hidden_dim)
        """
        # Save input for residual connection
        identity = x
        
        # Forward pass through block
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        
        # Add residual connection
        out += identity
        out = F.relu(out)
        
        return out


class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor for the JEPA model.
    Uses self-attention to predict target embeddings from context embeddings.
    Aligns with the I-JEPA approach for more powerful representation learning.
    """
    
    def __init__(self, embed_dim=256, output_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        """
        Initialize the transformer predictor.
        
        Args:
            embed_dim (int): Dimension of the input embeddings
            output_dim (int): Dimension of the output embeddings
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout (float): Dropout rate
        """
        super(TransformerPredictor, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass of the transformer predictor.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, embed_dim)
            
        Returns:
            Tensor: Predicted embeddings of shape (batch_size, output_dim)
        """
        # Reshape for transformer if needed
        if len(x.shape) == 2:
            # Add sequence length dimension
            x = x.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Take the output of the first token (like CLS token in BERT)
        x = x[:, 0]
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Adds information about the position of tokens in the sequence.
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model (int): The embedding dimension
            dropout (float): Dropout rate
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor: Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
