"""
Simplified training script for the Atari JEPA model.
Uses synthetic data to test the model pipeline.
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

# Import our simplified data processor
from custom_data.simplified_processor import prepare_simplified_data

# Import model components
from models.jepa import JEPA
from models.context_encoder import ContextEncoder, VisionTransformerEncoder
from models.predictor import Predictor, TransformerPredictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Atari JEPA model with synthetic data")
    
    # Data arguments
    parser.add_argument("--game_name", type=str, default="breakout", help="Name of the game to train on (e.g., breakout)")
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
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
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
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    return output_dir, device, writer


def create_model(args, device):
    """Create JEPA model."""
    # Create JEPA model with appropriate parameters
    model = JEPA(
        input_channels=1,  # Grayscale Atari frames
        action_dim=18,     # Standard Atari action space
        embedding_dim=args.embed_dim,
        hidden_dim=args.embed_dim * 2,
        context_encoder_type="resnet" if args.context_encoder == "vit" else "standard",
        target_encoder_type="embedding",
        predictor_type="residual" if args.predictor == "transformer" else "standard",
        temperature=0.07
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
            outputs = model(frames, actions)
            
            # Compute loss using the model's compute_loss method
            loss = model.compute_loss(outputs)
            
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
            outputs = model(frames, actions)
            
            # Compute loss using the model's compute_loss method
            loss = model.compute_loss(outputs)
            
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
    output_dir, device, writer = setup_training(args)
    
    # Prepare simplified data
    dataloaders = prepare_simplified_data(
        game_name=args.game_name,
        batch_size=args.batch_size
    )
    
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
