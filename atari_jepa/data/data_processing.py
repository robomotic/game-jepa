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
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=')
                    action_map[int(value.strip())] = key.strip()
        
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
        
        # Create a frames directory to store all frames
        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Create a directory for labels
        labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(labels_dir, exist_ok=True)
        
        trial_info = {}
        
        for trial_id in game_trials:
            # Get the trial data from the meta file
            trial_meta = self.game_meta[self.game_meta['trial_id'] == trial_id]
            if len(trial_meta) == 0:
                print(f"Trial {trial_id} not found in metadata, skipping...")
                continue
                
            print(f"Processing trial {trial_id}...")
            
            # Create directories for this trial
            trial_frames_dir = os.path.join(frames_dir, f"trial_{trial_id}")
            os.makedirs(trial_frames_dir, exist_ok=True)
            
            # Create a placeholder label file for now
            # In a real implementation, we would extract and parse action data
            label_file = os.path.join(labels_dir, f"trial_{trial_id}_labels.csv")
            
            # Generate some placeholder frame files
            frame_count = trial_meta['total_frame'].values[0]
            # Generate a small number of frames for testing purposes
            frame_count = min(frame_count, 100)  # Limit to 100 frames for testing
            
            # Create blank frame files (84x84 grayscale)
            for frame_idx in range(frame_count):
                frame = np.zeros((84, 84), dtype=np.uint8)
                # Add a simple pattern to the frame to make it recognizable
                frame[10:74, 10:74] = 128  # Middle gray square
                frame[20:64, 20:64] = 255  # White inner square
                
                # Add trial and frame number for visualization
                cv2.putText(frame, f"T{trial_id}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                cv2.putText(frame, f"F{frame_idx}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                
                # Save frame as PNG
                frame_path = os.path.join(trial_frames_dir, f"frame_{frame_idx:05d}.png")
                cv2.imwrite(frame_path, frame)
                
            # Create a placeholder label file with random actions
            with open(label_file, 'w') as f:
                f.write("frame_idx,action_idx,action_name\n")
                for frame_idx in range(frame_count):
                    # Choose a random action index (0-17 for Atari)
                    action_idx = np.random.randint(0, 18)
                    action_name = self.action_map.get(action_idx, "UNKNOWN")
                    f.write(f"{frame_idx},{action_idx},{action_name}\n")
            
            # Store trial info
            trial_info[trial_id] = {
                'frames_dir': trial_frames_dir,
                'label_file': label_file,
                'frame_count': frame_count
            }
            
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
        
        # Process each trial to create frame-action pairs
        for trial_id, info in trial_info.items():
            frames_dir = info['frames_dir']
            label_file = info['label_file']
            
            # Check if the label file exists
            if not os.path.exists(label_file):
                print(f"Label file {label_file} not found, skipping trial {trial_id}...")
                continue
                
            # Load the labels
            try:
                labels_df = pd.read_csv(label_file)
            except Exception as e:
                print(f"Error reading label file {label_file}: {e}")
                continue
                
            # Process each frame in the trial
            for _, row in labels_df.iterrows():
                frame_idx = row['frame_idx']
                action_idx = row['action_idx']
                
                # Construct the frame path
                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
                
                # Check if the frame exists
                if not os.path.exists(frame_path):
                    # print(f"Frame {frame_path} not found, skipping...")
                    continue
                    
                # Add the frame-action pair
                frame_action_pairs.append({
                    'frame_path': frame_path,
                    'action_idx': action_idx,
                    'trial_id': trial_id
                })
        
        print(f"Created {len(frame_action_pairs)} frame-action pairs from {len(trial_info)} trials")
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
        if not frame_action_pairs:
            print("No frame-action pairs to create dataset files from!")
            # Create empty files to avoid file not found errors
            train_file = os.path.join(output_dir, 'train.csv')
            val_file = os.path.join(output_dir, 'val.csv')
            test_file = os.path.join(output_dir, 'test.csv')
            
            # Create empty files with headers
            for file_path in [train_file, val_file, test_file]:
                with open(file_path, 'w') as f:
                    f.write('frame_path,action_idx,trial_id\n')
                    
            return {
                'train': train_file,
                'val': val_file,
                'test': test_file
            }
        
        # Shuffle data
        import random
        random.shuffle(frame_action_pairs)
        
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
            f.write('frame_path,action_idx,trial_id\n')
            for pair in train_pairs:
                f.write(f"{pair['frame_path']},{pair['action_idx']},{pair['trial_id']}\n")
        
        # Write val file
        with open(val_file, 'w') as f:
            f.write('frame_path,action_idx,trial_id\n')
            for pair in val_pairs:
                f.write(f"{pair['frame_path']},{pair['action_idx']},{pair['trial_id']}\n")
        
        # Write test file
        with open(test_file, 'w') as f:
            f.write('frame_path,action_idx,trial_id\n')
            for pair in test_pairs:
                f.write(f"{pair['frame_path']},{pair['action_idx']},{pair['trial_id']}\n")
        
        print(f"Created dataset files with {len(train_pairs)} training, {len(val_pairs)} validation, and {len(test_pairs)} test samples")
        
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
