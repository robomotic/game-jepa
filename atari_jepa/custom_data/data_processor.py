"""
Improved data processor for Atari-HEAD dataset.
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import zipfile
import tempfile

class AtariDataProcessor:
    """Process Atari-HEAD dataset for JEPA training."""
    
    def __init__(self, data_dir, game_name):
        """
        Initialize the data processor.
        
        Args:
            data_dir (str): Path to the Atari-HEAD dataset directory
            game_name (str): Name of the game to process (e.g., 'breakout')
        """
        self.data_dir = data_dir
        self.game_name = game_name
        self.zip_path = os.path.join(data_dir, f"{game_name}.zip")
        self.action_map = self._load_action_map()
        
    def _load_action_map(self):
        """
        Load action mapping from the action_enums.txt file.
        
        Returns:
            dict: Mapping from action enum to action name
        """
        action_map = {}
        action_file = os.path.join(self.data_dir, "action_enums.txt")
        
        with open(action_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                    
                # Parse lines like "PLAYER_A_NOOP = 0"
                match = re.match(r'(\w+)\s*=\s*(\d+)', line)
                if match:
                    action_name, action_id = match.groups()
                    action_map[int(action_id)] = action_name.strip()
        
        return action_map
    
    def extract_frames_and_actions(self, output_dir):
        """
        Extract frames and actions from the zip file.
        
        Args:
            output_dir (str): Directory to save processed data
            
        Returns:
            list: List of (frame_path, action_id) pairs
        """
        frame_action_pairs = []
        
        # Create temporary directory to extract files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract necessary files from the zip
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # Extract only necessary files (this is a placeholder - adjust based on actual zip structure)
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith('.png') or file_info.filename.endswith('.txt'):
                        zip_ref.extract(file_info, temp_dir)
            
            # Process the extracted files
            # This is a placeholder implementation - adjust based on actual data structure
            frame_dir = os.path.join(temp_dir, 'frames')
            action_file = os.path.join(temp_dir, 'actions.txt')
            
            # If the action file exists, process it along with frames
            if os.path.exists(action_file):
                with open(action_file, 'r') as f:
                    action_lines = f.readlines()
                
                # Process frames and match with actions
                for i, action_line in enumerate(action_lines):
                    action_id = int(action_line.strip())
                    frame_path = os.path.join(frame_dir, f"frame_{i:05d}.png")
                    
                    if os.path.exists(frame_path):
                        # Save to output directory
                        output_frame_path = os.path.join(output_dir, f"frame_{i:05d}.png")
                        os.makedirs(os.path.dirname(output_frame_path), exist_ok=True)
                        
                        # Copy frame to output directory
                        Image.open(frame_path).save(output_frame_path)
                        
                        frame_action_pairs.append((output_frame_path, action_id))
        
        return frame_action_pairs
    
    def create_dataset(self, output_dir):
        """
        Create a dataset from the Atari game data.
        
        Args:
            output_dir (str): Directory to save processed data
            
        Returns:
            AtariDataset: Dataset for training
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract frames and actions
        frame_action_pairs = self.extract_frames_and_actions(output_dir)
        
        # Create dataset
        dataset = AtariDataset(frame_action_pairs, self.action_map)
        
        return dataset


class AtariDataset(Dataset):
    """Dataset for Atari frames and actions."""
    
    def __init__(self, frame_action_pairs, action_map):
        """
        Initialize the dataset.
        
        Args:
            frame_action_pairs (list): List of (frame_path, action_id) pairs
            action_map (dict): Mapping from action ID to action name
        """
        self.frame_action_pairs = frame_action_pairs
        self.action_map = action_map
        
    def __len__(self):
        return len(self.frame_action_pairs)
    
    def __getitem__(self, idx):
        frame_path, action_id = self.frame_action_pairs[idx]
        
        # Load and preprocess frame
        frame = Image.open(frame_path).convert('L')  # Convert to grayscale
        frame = np.array(frame) / 255.0  # Normalize to [0, 1]
        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        # Convert action to one-hot encoding
        action = torch.zeros(len(self.action_map))
        action[action_id] = 1.0
        
        return frame, action


def create_dataloaders(dataset, batch_size=32, train_ratio=0.8):
    """
    Create train and validation dataloaders.
    
    Args:
        dataset (AtariDataset): The dataset to split
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
