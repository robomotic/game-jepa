"""
Training script for the Atari JEPA model with fixed data processing.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from tqdm import tqdm
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom data processor
from custom_data.data_processor import AtariDataProcessor, create_dataloaders

# Import model components
from models.jepa import JEPA
from models.context_encoder import ContextEncoder, VisionTransformerEncoder
from models.predictor import Predictor, TransformerPredictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Atari JEPA model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to Atari-HEAD dataset")
    parser.add_argument("--game_name", type=str, required=True, help="Name of the game to train on (e.g., breakout)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    
    # Model arguments
    parser.add_argument("--context_encoder", type=str, default="cnn", choices=["cnn", "vit"], 
                        help="Type of context encoder to use")
    parser.add_argument("--predictor", type=str, default="mlp", choices=["mlp", "transformer"], 
                        help="Type of predictor to use")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--use_masking", action="store_true", help="Use masking for self-supervised training")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for training")
    
    return parser.parse_args()


def setup_training(args):
    """Set up training environment."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"{args.game_name}_{args.context_encoder}_{args.predictor}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    
    # Create processed data directory
    processed_dir = os.path.join(output_dir, "processed_data")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    return output_dir, processed_dir, device, writer


def prepare_data(args, processed_dir):
    """Prepare data for training."""
    print("Preparing data...")
    
    # Create data processor
    processor = AtariDataProcessor(data_dir=args.data_path, game_name=args.game_name)
    
    # Create dataset
    dataset = processor.create_dataset(output_dir=processed_dir)
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        dataset, batch_size=args.batch_size, train_ratio=0.8
    )
    
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Train dataloader: {len(train_dataloader)} batches")
    print(f"Validation dataloader: {len(val_dataloader)} batches")
    
    return {"train": train_dataloader, "val": val_dataloader}


def create_model(args, device):
    """Create JEPA model."""
    # Determine input shape (assuming grayscale 84x84 Atari frames)
    input_shape = (1, 84, 84)
    
    # Create context encoder
    if args.context_encoder == "vit":
        context_encoder = VisionTransformerEncoder(
            img_size=84, 
            patch_size=7, 
            in_chans=1, 
            embed_dim=args.embed_dim, 
            depth=6, 
            num_heads=4
        )
    else:
        context_encoder = ContextEncoder(
            input_shape=input_shape, 
            embed_dim=args.embed_dim
        )
    
    # Create predictor
    if args.predictor == "transformer":
        predictor = TransformerPredictor(
            embed_dim=args.embed_dim, 
            output_dim=args.embed_dim, 
            num_heads=4, 
            num_layers=2
        )
    else:
        predictor = Predictor(
            input_dim=args.embed_dim, 
            output_dim=args.embed_dim
        )
    
    # Create JEPA model
    model = JEPA(
        context_encoder=context_encoder,
        predictor=predictor,
        use_masking=args.use_masking
    )
    
    return model.to(device)


def train_epoch(model, optimizer, dataloader, device, epoch):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    
    with tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch") as pbar:
        for frames, actions in pbar:
            # Move data to device
            frames = frames.to(device)
            actions = actions.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = model(frames, actions)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for frames, actions in dataloader:
            # Move data to device
            frames = frames.to(device)
            actions = actions.to(device)
            
            # Forward pass
            loss = model(frames, actions)
            
            # Accumulate loss
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_model(args, model, dataloaders, optimizer, output_dir, device, writer):
    """Train the model."""
    print("Starting training...")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=dataloaders["train"],
            device=device,
            epoch=epoch
        )
        
        # Validate
        val_loss = validate(
            model=model,
            dataloader=dataloaders["val"],
            device=device
        )
        
        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "args": vars(args)
            }
            
            torch.save(checkpoint, os.path.join(output_dir, "best_model.pth"))
            print(f"Saved best model checkpoint (Epoch {epoch+1})")
    
    # Save final model
    checkpoint = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "args": vars(args)
    }
    
    torch.save(checkpoint, os.path.join(output_dir, "final_model.pth"))
    print(f"Saved final model checkpoint (Epoch {args.epochs})")
    
    return model


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Setup training
    output_dir, processed_dir, device, writer = setup_training(args)
    
    # Prepare data
    dataloaders = prepare_data(args, processed_dir)
    
    # Create model
    model = create_model(args, device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Train model
    trained_model = train_model(
        args=args,
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        output_dir=output_dir,
        device=device,
        writer=writer
    )
    
    # Close TensorBoard writer
    writer.close()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
