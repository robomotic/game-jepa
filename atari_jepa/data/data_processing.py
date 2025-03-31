"""
Data processing utilities for the Atari-HEAD dataset.
Handles extraction of frames, actions, and metadata from the dataset.
"""

import os
import tarfile
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import io
import re


class AtariDataProcessor:
    """
    Process the Atari-HEAD dataset for Breakout game.
    Extracts frames and action data from the dataset files.
    """
    
    def __init__(self, data_dir, game_name="breakout"):
        """
        Initialize the data processor.
        
        Args:
            data_dir (str): Path to the Atari directory containing the game data
            game_name (str): Name of the game to process (default: "breakout")
        """
        self.data_dir = data_dir
        self.game_name = game_name
        self.meta_data = pd.read_csv(os.path.join(data_dir, "meta_data.csv"))
        self.game_meta = self.meta_data[self.meta_data['GameName'] == game_name]
        
        # Load action mappings
        self.action_map = self._load_action_map()
        
    def _load_action_map(self):
        """Load action enumeration mappings"""
        action_file = os.path.join(self.data_dir, "action_enums.txt")
        action_map = {}
        
        with open(action_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=')
                    action_map[int(key)] = value.strip()
        
        return action_map
    
    def extract_trials(self, output_dir, max_trials=None):
        """
        Extract trial data (frames and actions) for the specified game.
        
        Args:
            output_dir (str): Directory to save extracted data
            max_trials (int, optional): Maximum number of trials to extract
        
        Returns:
            dict: Mapping of trial IDs to extracted data info
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter trials for the game
        game_trials = self.game_meta['trial_id'].unique()
        if max_trials:
            game_trials = game_trials[:max_trials]
        
        trial_info = {}
        
        for trial_id in game_trials:
            trial_path = os.path.join(self.data_dir, f"{self.game_name}.zip")
            
            # Extract data for this trial
            trial_output = os.path.join(output_dir, f"trial_{trial_id}")
            os.makedirs(trial_output, exist_ok=True)
            
            print(f"Processing trial {trial_id}...")
            
            # Here we would extract the trial data from the zip file
            # For now we'll just create a placeholder for the structure
            trial_info[trial_id] = {
                'frames_dir': os.path.join(trial_output, 'frames'),
                'label_file': os.path.join(trial_output, f"trial_{trial_id}_labels.csv"),
                'frame_count': self.game_meta[self.game_meta['trial_id'] == trial_id]['total_frame'].values[0]
            }
            
            # In real implementation, we would extract frames from tar.bz2 and parse label file
            # This is a placeholder for the actual extraction logic
            
        return trial_info
    
    def extract_frame_action_pairs(self, trial_info):
        """
        Extract frame-action pairs from the trial data.
        
        Args:
            trial_info (dict): Trial information from extract_trials
            
        Returns:
            list: List of (frame_path, action) pairs
        """
        frame_action_pairs = []
        
        # Placeholder for the actual extraction logic
        # In a real implementation, we would:
        # 1. Extract frames from the tar.bz2 files
        # 2. Parse the label files to get actions for each frame
        # 3. Create pairs of (frame_path, action)
        
        return frame_action_pairs
    
    def preprocess_frame(self, frame, target_size=(84, 84)):
        """
        Preprocess a game frame for input to the model.
        
        Args:
            frame (PIL.Image): Input frame
            target_size (tuple): Target size for resizing
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        # Convert to grayscale
        if frame.mode != 'L':
            frame = frame.convert('L')
        
        # Resize
        frame = frame.resize(target_size, Image.BILINEAR)
        
        # Convert to numpy array and normalize
        frame_array = np.array(frame, dtype=np.float32) / 255.0
        
        return frame_array
    
    def create_dataset_files(self, output_dir, frame_action_pairs, split=(0.7, 0.15, 0.15)):
        """
        Create train/val/test dataset files.
        
        Args:
            output_dir (str): Output directory
            frame_action_pairs (list): List of (frame_path, action) pairs
            split (tuple): Train/val/test split ratios
            
        Returns:
            dict: Paths to the dataset files
        """
        # Shuffle data
        np.random.shuffle(frame_action_pairs)
        
        # Split data
        n_samples = len(frame_action_pairs)
        n_train = int(n_samples * split[0])
        n_val = int(n_samples * split[1])
        
        train_pairs = frame_action_pairs[:n_train]
        val_pairs = frame_action_pairs[n_train:n_train+n_val]
        test_pairs = frame_action_pairs[n_train+n_val:]
        
        # Create dataset files
        train_file = os.path.join(output_dir, 'train.csv')
        val_file = os.path.join(output_dir, 'val.csv')
        test_file = os.path.join(output_dir, 'test.csv')
        
        # Write train file
        with open(train_file, 'w') as f:
            f.write('frame_path,action\n')
            for frame_path, action in train_pairs:
                f.write(f"{frame_path},{action}\n")
        
        # Write val file
        with open(val_file, 'w') as f:
            f.write('frame_path,action\n')
            for frame_path, action in val_pairs:
                f.write(f"{frame_path},{action}\n")
        
        # Write test file
        with open(test_file, 'w') as f:
            f.write('frame_path,action\n')
            for frame_path, action in test_pairs:
                f.write(f"{frame_path},{action}\n")
        
        return {
            'train': train_file,
            'val': val_file,
            'test': test_file
        }


if __name__ == "__main__":
    # Example usage
    processor = AtariDataProcessor(data_dir="/media/robomotic/bumbledisk/github/game-jepa/Atari")
    trial_info = processor.extract_trials(output_dir="/media/robomotic/bumbledisk/github/game-jepa/atari_jepa/data/processed")
    frame_action_pairs = processor.extract_frame_action_pairs(trial_info)
    dataset_files = processor.create_dataset_files(
        output_dir="/media/robomotic/bumbledisk/github/game-jepa/atari_jepa/data/processed", 
        frame_action_pairs=frame_action_pairs
    )
    
    print(f"Created dataset files: {dataset_files}")
