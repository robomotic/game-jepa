"""
Combined training script for the Atari JEPA model.
Supports multiple data processing modes and model architectures.
"""

import os
import sys
import argparse
import time
import json
import random
from datetime import datetime
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model components
from models.jepa import JEPA
from models.context_encoder import ContextEncoder, ResNetContextEncoder, VisionTransformerEncoder
from models.target_encoder import TargetEncoder, AtariActionEncoder
from models.predictor import Predictor, ResidualPredictor, TransformerPredictor


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train the Atari JEPA model')
    
    # Data arguments
    parser.add_argument('--data_mode', type=str, default='real',
                       choices=['real', 'synthetic', 'fixed'],
                       help='Data processing mode to use (real, synthetic, or fixed)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to the Atari-HEAD dataset (required for real and fixed modes)')
    parser.add_argument('--game_name', type=str, default='breakout',
                       help='Name of the game to train on')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                       help='Directory to save outputs')
    parser.add_argument('--synthetic_samples', type=int, default=1000,
                       help='Number of synthetic samples to generate (synthetic mode only)')
    
    # Model arguments
    parser.add_argument('--context_encoder', type=str, default='cnn',
                       choices=['cnn', 'resnet', 'vit'],
                       help='Type of context encoder')
    parser.add_argument('--target_encoder', type=str, default='embedding',
                       choices=['standard', 'embedding'],
                       help='Type of target encoder')
    parser.add_argument('--predictor', type=str, default='mlp',
                       choices=['mlp', 'residual', 'transformer'],
                       help='Type of predictor')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Dimension of embeddings')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Dimension of hidden layers')
    parser.add_argument('--use_masking', action='store_true',
                       help='Use masking for self-supervised training')
    parser.add_argument('--mask_ratio', type=float, default=0.4,
                       help='Ratio of tokens/patches to mask during training (if use_masking is enabled)')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Interval to save model checkpoints')
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='Interval to evaluate model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    
    args = parser.parse_args()
    
    # Validate arguments based on data mode
    if args.data_mode in ['real', 'fixed'] and args.data_path is None:
        parser.error("--data_path is required for 'real' and 'fixed' data modes")
    
    return args


def setup_training(args):
    """Set up training environment."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"{args.game_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    return output_dir, device, writer


def prepare_data(args):
    """Prepare data for training based on the specified mode."""
    print(f"Preparing data using '{args.data_mode}' mode...")
    
    # Create processed data directory if needed
    processed_dir = os.path.join(args.output_dir, 'processed_data')
    os.makedirs(processed_dir, exist_ok=True)
    
    if args.data_mode == 'synthetic':
        # Use synthetic data
        from custom_data.simplified_processor import prepare_simplified_data
        dataloaders = prepare_simplified_data(
            game_name=args.game_name,
            batch_size=args.batch_size,
            num_samples=args.synthetic_samples
        )
    
    elif args.data_mode == 'fixed':
        # Use fixed data processor
        from custom_data.data_processor import AtariDataProcessor, create_dataloaders
        
        # Initialize data processor
        processor = AtariDataProcessor(data_dir=args.data_path, game_name=args.game_name)
        
        # Extract frames and actions
        print("Extracting frames and actions...")
        frame_action_pairs = processor.extract_frames_and_actions(processed_dir)
        
        # Create dataset and dataloaders
        from custom_data.data_processor import AtariDataset
        dataset = AtariDataset(frame_action_pairs, processor.action_map)
        train_loader, val_loader = create_dataloaders(
            dataset=dataset,
            batch_size=args.batch_size,
            train_ratio=0.8
        )
        
        dataloaders = {
            'train': train_loader,
            'val': val_loader,
            'test': val_loader  # Use validation as test for now
        }
    
    else:  # args.data_mode == 'real'
        # Use standard data processor
        from data.data_processing import AtariDataProcessor
        from data.data_loader import create_dataloaders
        
        # Create data processor
        processor = AtariDataProcessor(data_dir=args.data_path, game_name=args.game_name)
        
        # Extract trial data
        print("Extracting trial data...")
        trial_info = processor.extract_trials(output_dir=processed_dir)
        
        print("Creating frame-action pairs...")
        frame_action_pairs = processor.extract_frame_action_pairs(trial_info)
        
        print("Creating dataset files...")
        dataset_files = processor.create_dataset_files(
            output_dir=processed_dir, 
            frame_action_pairs=frame_action_pairs
        )
        
        # Create data loaders
        dataloaders = create_dataloaders(
            train_csv=dataset_files['train'],
            val_csv=dataset_files['val'],
            test_csv=dataset_files['test'],
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
    return dataloaders


def create_model(args, device):
    """Create JEPA model with the specified architecture."""
    print("Creating model...")
    
    # Set up context encoder based on args
    if args.context_encoder == 'cnn':
        context_encoder_type = 'standard'
    elif args.context_encoder == 'resnet':
        context_encoder_type = 'resnet'
    elif args.context_encoder == 'vit':
        context_encoder_type = 'vit'
    else:
        context_encoder_type = 'standard'
    
    # Set up predictor based on args
    if args.predictor == 'mlp':
        predictor_type = 'standard'
    elif args.predictor == 'residual':
        predictor_type = 'residual'
    elif args.predictor == 'transformer':
        predictor_type = 'transformer'
    else:
        predictor_type = 'standard'
    
    # Create model
    model = JEPA(
        input_channels=1,  # Grayscale images
        action_dim=18,  # Number of Atari actions
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        context_encoder_type=context_encoder_type,
        target_encoder_type=args.target_encoder,
        predictor_type=predictor_type,
        temperature=args.temperature,
        use_masking=args.use_masking,
        mask_ratio=args.mask_ratio
    )
    
    # Move model to device
    model = model.to(device)
    
    return model


def train_epoch(model, optimizer, dataloader, device, epoch):
    """Train model for one epoch."""
    model.train()
    
    epoch_loss = 0.0
    num_batches = len(dataloader)
    start_time = time.time()
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{model.epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        # Get batch data - handle different data formats
        if isinstance(batch, dict):
            # Format from AtariFrameActionDataset
            frames = batch['frame'].to(device)
            actions = batch.get('action', None)
            if actions is not None:
                actions = actions.to(device)
            action_idx = batch.get('action_idx', None)
            if action_idx is not None:
                action_idx = action_idx.to(device)
        else:
            # Format from SimplifiedAtariDataset
            frames, action_idx = batch
            frames = frames.to(device)
            action_idx = action_idx.to(device)
            actions = None
        
        # Forward pass with optional masking
        outputs = model(frames, actions, action_idx, apply_mask=model.use_masking)
        
        # Compute loss
        loss = model.compute_loss(outputs)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update epoch loss
        epoch_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})
    
    # Calculate average epoch loss
    epoch_loss = epoch_loss / num_batches
    
    return epoch_loss


def validate(model, dataloader, device):
    """Validate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    reciprocal_ranks = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get batch data - handle different data formats
            if isinstance(batch, dict):
                # Format from AtariFrameActionDataset
                frames = batch['frame'].to(device)
                actions = batch.get('action', None)
                if actions is not None:
                    actions = actions.to(device)
                action_idx = batch.get('action_idx', None)
                if action_idx is not None:
                    action_idx = action_idx.to(device)
            else:
                # Format from SimplifiedAtariDataset
                frames, action_idx = batch
                frames = frames.to(device)
                action_idx = action_idx.to(device)
                actions = None
            
            # Forward pass with masking disabled during evaluation
            outputs = model(frames, actions, action_idx, apply_mask=False)
            
            # Compute loss
            loss = model.compute_loss(outputs)
            
            # Update total loss
            total_loss += loss.item()
            
            # Compute accuracy and MRR
            similarity = model.compute_similarity(
                outputs['predicted_target_embeddings'],
                outputs['target_embeddings']
            )
            
            # Get predictions
            _, predicted = similarity.max(1)
            
            # Get ground truth
            # For contrastive learning, the ground truth is the diagonal elements
            targets = torch.arange(similarity.size(0), device=device)
            
            # Update correct predictions
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # Compute reciprocal rank
            for i, target in enumerate(targets):
                # Get rank of the target (1-based)
                rank = (similarity[i].argsort(descending=True) == target).nonzero().item() + 1
                reciprocal_ranks.append(1.0 / rank)
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    
    return avg_loss, accuracy, mrr


def train_model(args, model, dataloaders, optimizer, output_dir, device, writer):
    """Train the model."""
    print("Starting training...")
    
    best_val_loss = float('inf')
    best_accuracy = 0.0
    
    # Store epochs in model for access in train_epoch
    model.epochs = args.epochs
    
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(model, optimizer, dataloaders['train'], device, epoch)
        
        # Log training loss
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validate model
        if epoch % args.eval_interval == 0:
            print("Validating model...")
            val_loss, accuracy, mrr = validate(model, dataloaders['val'], device)
            
            # Log validation metrics
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', accuracy, epoch)
            writer.add_scalar('MRR/val', mrr, epoch)
            
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"MRR: {mrr:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'accuracy': accuracy,
                    'mrr': mrr
                }, best_model_path)
                print(f"Saved best model to {best_model_path}")
            
            # Save best accuracy model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_acc_model_path = os.path.join(output_dir, 'checkpoints', 'best_accuracy_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'accuracy': accuracy,
                    'mrr': mrr
                }, best_acc_model_path)
                print(f"Saved best accuracy model to {best_acc_model_path}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoints', f'model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final evaluation
    print("Final evaluation...")
    test_loss, test_accuracy, test_mrr = validate(model, dataloaders['test'], device)
    
    # Log test metrics
    writer.add_scalar('Loss/test', test_loss, 0)
    writer.add_scalar('Accuracy/test', test_accuracy, 0)
    writer.add_scalar('MRR/test', test_mrr, 0)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test MRR: {test_mrr:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'checkpoints', 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_mrr': test_mrr
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up training environment
    output_dir, device, writer = setup_training(args)
    
    # Prepare data
    dataloaders = prepare_data(args)
    
    # Create model
    model = create_model(args, device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Train model
    train_model(args, model, dataloaders, optimizer, output_dir, device, writer)
    
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
