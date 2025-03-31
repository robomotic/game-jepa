"""
Data loaders for the Atari-HEAD dataset.
Provides PyTorch dataset classes for loading frame-action pairs.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random


class AtariFrameActionDataset(Dataset):
    """
    PyTorch dataset for loading frame-action pairs from the Atari-HEAD dataset.
    Used for training and evaluating the JEPA model.
    """
    
    def __init__(self, csv_file, transform=None, action_dim=18):
        """
        Initialize the dataset.
        
        Args:
            csv_file (str): Path to the CSV file containing frame paths and actions
            transform (callable, optional): Optional transform to be applied on a sample
            action_dim (int): Dimension of the action space (default: 18 for Atari)
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.action_dim = action_dim
        
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample containing frame and action
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get frame path and action
        frame_path = self.data.iloc[idx, 0]
        action = self.data.iloc[idx, 1]
        
        # Load frame
        frame = Image.open(frame_path)
        
        # Apply transform if provided
        if self.transform:
            frame = self.transform(frame)
        else:
            # Default processing - convert to tensor and normalize
            frame = np.array(frame, dtype=np.float32) / 255.0
            frame = torch.from_numpy(frame).unsqueeze(0)  # Add channel dimension
        
        # Convert action to one-hot encoding
        action_tensor = torch.zeros(self.action_dim)
        action_tensor[action] = 1.0
        
        return {
            'frame': frame,
            'action': action_tensor,
            'action_idx': torch.tensor(action, dtype=torch.long)
        }


class JEPABatchSampler:
    """
    Batch sampler for JEPA training.
    Creates positive and negative pairs for contrastive learning.
    """
    
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        Initialize the batch sampler.
        
        Args:
            dataset (Dataset): Dataset to sample from
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle the dataset
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
    def __iter__(self):
        """
        Iterate over batches.
        
        Returns:
            Iterator: Batch indices
        """
        if self.shuffle:
            random.shuffle(self.indices)
            
        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i:i + self.batch_size]
            
    def __len__(self):
        """Return the number of batches"""
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def create_dataloaders(train_csv, val_csv, test_csv, batch_size=32, num_workers=4):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_csv (str): Path to the training CSV file
        val_csv (str): Path to the validation CSV file
        test_csv (str): Path to the test CSV file
        batch_size (int): Batch size
        num_workers (int): Number of worker threads
        
    Returns:
        dict: Data loaders for training, validation, and testing
    """
    # Create datasets
    train_dataset = AtariFrameActionDataset(train_csv)
    val_dataset = AtariFrameActionDataset(val_csv)
    test_dataset = AtariFrameActionDataset(test_csv)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    # Example usage
    data_dir = "/media/robomotic/bumbledisk/github/game-jepa/atari_jepa/data/processed"
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    
    dataloaders = create_dataloaders(train_csv, val_csv, test_csv)
    
    # Test train loader
    for batch in dataloaders['train']:
        frames = batch['frame']
        actions = batch['action']
        print(f"Batch shapes: frames={frames.shape}, actions={actions.shape}")
        break
