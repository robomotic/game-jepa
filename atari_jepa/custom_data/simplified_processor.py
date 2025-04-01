"""
Simplified data processor for Atari-HEAD dataset.
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
import zipfile
import tempfile
import tarfile
import bz2
import io
import shutil

class SimplifiedAtariDataset(Dataset):
    """
    A simplified dataset for Atari-HEAD that creates synthetic data for testing.
    This allows us to test the model pipeline without needing to extract the complex
    Atari-HEAD dataset structure.
    """
    
    def __init__(self, game_name, num_samples=1000, img_size=84, num_actions=18):
        """
        Initialize the simplified dataset.
        
        Args:
            game_name (str): Name of the game (for logging purposes)
            num_samples (int): Number of synthetic samples to generate
            img_size (int): Size of the synthetic frames
            num_actions (int): Number of possible actions
        """
        self.game_name = game_name
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_actions = num_actions
        
        print(f"Creating simplified dataset for {game_name} with {num_samples} samples")
        
        # Generate synthetic data
        self.frames = torch.rand(num_samples, 1, img_size, img_size)  # Random frames
        
        # Generate action indices (not one-hot encoded)
        self.actions = torch.randint(0, num_actions, (num_samples,), dtype=torch.long)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.frames[idx], self.actions[idx]


def create_dataloaders(dataset, batch_size=32, train_ratio=0.8):
    """
    Create train and validation dataloaders.
    
    Args:
        dataset (Dataset): The dataset to split
        batch_size (int): Batch size for dataloaders
        train_ratio (float): Ratio of data to use for training
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    # Split dataset into train and validation
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader


def prepare_simplified_data(game_name, batch_size=32, num_samples=1000):
    """
    Prepare simplified synthetic data for testing the model pipeline.
    
    Args:
        game_name (str): Name of the game
        batch_size (int): Batch size for dataloaders
        
    Returns:
        dict: Dictionary containing train and validation dataloaders
    """
    # Create simplified dataset
    dataset = SimplifiedAtariDataset(game_name=game_name, num_samples=num_samples)
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        dataset, batch_size=batch_size, train_ratio=0.8
    )
    
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Train dataloader: {len(train_dataloader)} batches")
    print(f"Validation dataloader: {len(val_dataloader)} batches")
    
    # Use validation dataloader as test dataloader as well
    return {"train": train_dataloader, "val": val_dataloader, "test": val_dataloader}
