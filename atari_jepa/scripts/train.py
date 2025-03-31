"""
Training script for the Atari JEPA model.
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processing import AtariDataProcessor
from data.data_loader import AtariFrameActionDataset, create_dataloaders
from models.jepa import JEPA
from utils.metrics import evaluate_jepa_model
from utils.visualization import save_visualizations


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train the Atari JEPA model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='/media/robomotic/bumbledisk/github/game-jepa/Atari',
                       help='Path to the Atari-HEAD dataset')
    parser.add_argument('--game_name', type=str, default='breakout',
                       help='Name of the game to train on')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                       help='Directory to save outputs')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Dimension of embeddings')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Dimension of hidden layers')
    parser.add_argument('--context_encoder', type=str, default='resnet',
                       choices=['standard', 'resnet'],
                       help='Type of context encoder')
    parser.add_argument('--target_encoder', type=str, default='embedding',
                       choices=['standard', 'embedding'],
                       help='Type of target encoder')
    parser.add_argument('--predictor', type=str, default='standard',
                       choices=['standard', 'residual'],
                       help='Type of predictor')
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
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    
    return parser.parse_args()


def setup_training(args):
    """Set up training environment."""
    # Create output directory
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
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    return output_dir, device, writer


def prepare_data(args, processed_dir):
    """Prepare data for training."""
    print("Preparing data...")
    
    # Create data processor
    processor = AtariDataProcessor(data_dir=args.data_path, game_name=args.game_name)
    
    # Extract trial data (in real implementation)
    # This is a placeholder - in a real implementation, we would extract the data here
    # trial_info = processor.extract_trials(output_dir=processed_dir)
    # frame_action_pairs = processor.extract_frame_action_pairs(trial_info)
    # dataset_files = processor.create_dataset_files(output_dir=processed_dir, frame_action_pairs=frame_action_pairs)
    
    # For this skeleton project, we'll assume the data is already processed
    # and the dataset files exist
    dataset_files = {
        'train': os.path.join(processed_dir, 'train.csv'),
        'val': os.path.join(processed_dir, 'val.csv'),
        'test': os.path.join(processed_dir, 'test.csv')
    }
    
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
    """Create JEPA model."""
    print("Creating model...")
    
    # Create model
    model = JEPA(
        input_channels=1,  # Grayscale images
        action_dim=18,  # Number of Atari actions
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        context_encoder_type=args.context_encoder,
        target_encoder_type=args.target_encoder,
        predictor_type=args.predictor,
        temperature=args.temperature
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
    
    for batch_idx, batch in enumerate(dataloader):
        # Get batch data
        frames = batch['frame'].to(device)
        action_idx = batch['action_idx'].to(device)
        actions = batch['action'].to(device)
        
        # Forward pass
        outputs = model(frames, actions, action_idx)
        
        # Compute loss
        loss = model.compute_loss(outputs)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update epoch loss
        epoch_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch} [{batch_idx+1}/{num_batches}] - "
                  f"Loss: {loss.item():.4f} - "
                  f"Time: {elapsed:.2f}s")
    
    # Calculate average epoch loss
    epoch_loss = epoch_loss / num_batches
    
    return epoch_loss


def validate(model, dataloader, device):
    """Validate model on validation set."""
    model.eval()
    
    val_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            # Get batch data
            frames = batch['frame'].to(device)
            action_idx = batch['action_idx'].to(device)
            actions = batch['action'].to(device)
            
            # Forward pass
            outputs = model(frames, actions, action_idx)
            
            # Compute loss
            loss = model.compute_loss(outputs)
            
            # Update validation loss
            val_loss += loss.item()
    
    # Calculate average validation loss
    val_loss = val_loss / num_batches
    
    return val_loss


def train_model(args, model, dataloaders, optimizer, output_dir, device, writer):
    """Train the model."""
    print("Starting training...")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Train one epoch
        train_loss = train_epoch(model, optimizer, dataloaders['train'], device, epoch)
        
        # Log training loss
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validate model
        if epoch % args.eval_interval == 0:
            print("Validating model...")
            val_loss = validate(model, dataloaders['val'], device)
            
            # Log validation loss
            writer.add_scalar('Loss/val', val_loss, epoch)
            
            # Evaluate model metrics
            metrics = evaluate_jepa_model(model, dataloaders['val'], device)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
            
            # Print metrics
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"MRR: {metrics['MRR']:.4f}")
            
            # Save visualizations
            save_visualizations(
                out_path=os.path.join(output_dir, 'visualizations'),
                epoch=epoch,
                model=model,
                dataloader=dataloaders['val'],
                device=device
            )
            
            # Save model if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': metrics
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
        
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
    
    # Save final model
    final_path = os.path.join(output_dir, 'checkpoints', 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    print("Training completed!")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up training environment
    output_dir, device, writer = setup_training(args)
    
    # Create processed data directory
    processed_dir = os.path.join(args.output_dir, 'processed_data')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Prepare data
    dataloaders = prepare_data(args, processed_dir)
    
    # Create model
    model = create_model(args, device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Train model
    train_model(args, model, dataloaders, optimizer, output_dir, device, writer)
    
    # Close TensorBoard writer
    writer.close()


if __name__ == '__main__':
    main()
