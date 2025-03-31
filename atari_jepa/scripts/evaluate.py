"""
Evaluation script for the Atari JEPA model.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import AtariFrameActionDataset, create_dataloaders
from models.jepa import JEPA
from utils.metrics import evaluate_jepa_model, compute_top_k_accuracy, compute_confusion_matrix
from utils.visualization import (
    plot_frame, plot_action_distribution, plot_embeddings_tsne,
    plot_similarity_matrix, visualize_prediction, visualize_batch
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate the Atari JEPA model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to the model configuration file (args.json)')
    
    # Data arguments
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to the test CSV file')
    parser.add_argument('--output_dir', type=str, default='../evaluation_results',
                       help='Directory to save evaluation results')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to evaluate on')
    parser.add_argument('--visualize_samples', type=int, default=10,
                       help='Number of samples to visualize')
    
    return parser.parse_args()


def load_model(args):
    """Load the trained model."""
    # Load model configuration
    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        # Try to find config in the same directory as the model
        config_path = os.path.join(os.path.dirname(args.model_path), 'args.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                'embedding_dim': 256,
                'hidden_dim': 512,
                'context_encoder': 'resnet',
                'target_encoder': 'embedding',
                'predictor': 'standard',
                'temperature': 0.07
            }
    
    # Create model
    model = JEPA(
        input_channels=1,  # Grayscale images
        action_dim=18,  # Number of Atari actions
        embedding_dim=config.get('embedding_dim', 256),
        hidden_dim=config.get('hidden_dim', 512),
        context_encoder_type=config.get('context_encoder', 'resnet'),
        target_encoder_type=config.get('target_encoder', 'embedding'),
        predictor_type=config.get('predictor', 'standard'),
        temperature=config.get('temperature', 0.07)
    )
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    return model, config


def prepare_data(args):
    """Prepare data for evaluation."""
    # Create test dataset
    test_dataset = AtariFrameActionDataset(args.test_csv)
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return test_loader


def evaluate(model, dataloader, device, output_dir, visualize_samples=10):
    """Evaluate the model and save results."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_jepa_model(model, dataloader, device)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print metrics
    print("Evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Get a batch for visualization
    batch = next(iter(dataloader))
    frames = batch['frame'].to(device)
    action_idx = batch['action_idx'].to(device)
    actions = batch['action'].to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(frames, actions, action_idx)
        action_probs = model.predict_action(frames)
        pred_actions = torch.argmax(action_probs, dim=1)
    
    # Compute similarity matrix
    similarity = model.compute_similarity(
        outputs['predicted_target_embeddings'], 
        outputs['target_embeddings']
    )
    
    # Visualize predictions
    n_samples = min(visualize_samples, frames.size(0))
    
    # Create figure for prediction visualization
    plt.figure(figsize=(12, n_samples * 3))
    for i in range(n_samples):
        plt.subplot(n_samples, 2, i * 2 + 1)
        plot_frame(frames[i].cpu(), title=f"Frame {i}")
        
        plt.subplot(n_samples, 2, i * 2 + 2)
        plot_action_distribution(action_probs[i].cpu(), title=f"Action Probs (True: {action_idx[i].item()}, Pred: {pred_actions[i].item()})")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_visualization.png'), dpi=300)
    plt.close()
    
    # Visualize embeddings with t-SNE
    print("Visualizing embeddings with t-SNE...")
    plt.figure(figsize=(10, 8))
    plot_embeddings_tsne(
        outputs['context_embeddings'].cpu(),
        action_idx.cpu(),
        title="Frame Embeddings Colored by Action"
    )
    plt.savefig(os.path.join(output_dir, 'frame_embeddings_tsne.png'), dpi=300)
    plt.close()
    
    # Visualize similarity matrix
    plt.figure(figsize=(8, 8))
    plot_similarity_matrix(
        similarity.cpu(),
        title="Similarity Matrix (Predicted vs. Target Embeddings)"
    )
    plt.savefig(os.path.join(output_dir, 'similarity_matrix.png'), dpi=300)
    plt.close()
    
    # Compute confusion matrix
    confusion = compute_confusion_matrix(pred_actions.cpu(), action_idx.cpu(), normalize='true')
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Compute top-k accuracy
    top_k_values = [1, 3, 5]
    top_k_results = {}
    for k in top_k_values:
        top_k_acc = compute_top_k_accuracy(action_probs.cpu(), action_idx.cpu(), k=k)
        top_k_results[f'top_{k}_accuracy'] = top_k_acc
    
    # Save top-k results
    with open(os.path.join(output_dir, 'top_k_accuracy.json'), 'w') as f:
        json.dump(top_k_results, f, indent=4)
    
    # Print top-k results
    print("Top-k accuracy:")
    for k, acc in top_k_results.items():
        print(f"  {k}: {acc:.4f}")
    
    print(f"Evaluation results saved to {output_dir}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Load model
    model, config = load_model(args)
    
    # Prepare data
    dataloader = prepare_data(args)
    
    # Evaluate model
    evaluate(model, dataloader, device, args.output_dir, args.visualize_samples)


if __name__ == '__main__':
    main()
