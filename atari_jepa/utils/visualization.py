"""
Visualization utilities for the Atari JEPA model.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE


def plot_frame(frame, ax=None, title=None):
    """
    Plot a single frame.
    
    Args:
        frame (numpy.ndarray): Frame to plot
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on
        title (str, optional): Title for the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Ensure frame is a numpy array
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    
    # If frame has a channel dimension and it's 1, squeeze it
    if frame.shape[0] == 1:
        frame = frame.squeeze(0)
    
    # If frame has values in [0, 1], scale to [0, 255]
    if frame.max() <= 1.0:
        frame = frame * 255
    
    ax.imshow(frame, cmap='gray')
    ax.axis('off')
    
    if title:
        ax.set_title(title)


def plot_action_distribution(action_probs, action_names=None, ax=None, title=None):
    """
    Plot action probabilities.
    
    Args:
        action_probs (numpy.ndarray): Action probabilities
        action_names (list, optional): List of action names
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on
        title (str, optional): Title for the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ensure action_probs is a numpy array
    if isinstance(action_probs, torch.Tensor):
        action_probs = action_probs.cpu().numpy()
    
    # If action_names is not provided, create generic names
    if action_names is None:
        action_names = [f"Action {i}" for i in range(len(action_probs))]
    
    # Plot action probabilities
    ax.bar(action_names, action_probs)
    ax.set_xlabel('Action')
    ax.set_ylabel('Probability')
    ax.set_ylim([0, 1])
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()


def plot_embeddings_tsne(embeddings, labels=None, ax=None, title=None, alpha=0.7):
    """
    Plot embeddings using t-SNE.
    
    Args:
        embeddings (numpy.ndarray): Embeddings to plot
        labels (numpy.ndarray, optional): Labels for coloring points
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on
        title (str, optional): Title for the plot
        alpha (float): Alpha value for points
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Ensure embeddings is a numpy array
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot embeddings
    if labels is not None:
        # Ensure labels is a numpy array
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Get unique labels and assign colors
        unique_labels = np.unique(labels)
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each class with a different color
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                color=colors[i],
                alpha=alpha,
                label=f"Action {label}"
            )
        
        ax.legend()
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=alpha)
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()


def plot_similarity_matrix(similarity_matrix, ax=None, title=None):
    """
    Plot a similarity matrix.
    
    Args:
        similarity_matrix (numpy.ndarray): Similarity matrix to plot
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on
        title (str, optional): Title for the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Ensure similarity_matrix is a numpy array
    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.cpu().numpy()
    
    # Plot similarity matrix
    im = ax.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(im, ax=ax)
    
    ax.set_xlabel('Target Embeddings')
    ax.set_ylabel('Predicted Embeddings')
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()


def plot_training_curves(train_losses, val_losses=None, train_accs=None, val_accs=None):
    """
    Plot training and validation curves.
    
    Args:
        train_losses (list): Training losses
        val_losses (list, optional): Validation losses
        train_accs (list, optional): Training accuracies
        val_accs (list, optional): Validation accuracies
    """
    n_epochs = len(train_losses)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    axes[0].plot(range(1, n_epochs + 1), train_losses, label='Train Loss')
    if val_losses:
        axes[0].plot(range(1, n_epochs + 1), val_losses, label='Val Loss')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracies if provided
    if train_accs or val_accs:
        if train_accs:
            axes[1].plot(range(1, n_epochs + 1), train_accs, label='Train Acc')
        if val_accs:
            axes[1].plot(range(1, n_epochs + 1), val_accs, label='Val Acc')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    

def visualize_prediction(frame, true_action, pred_action_probs, action_names=None):
    """
    Visualize a frame with its true action and predicted action probabilities.
    
    Args:
        frame (numpy.ndarray): Frame to visualize
        true_action (int): True action index
        pred_action_probs (numpy.ndarray): Predicted action probabilities
        action_names (list, optional): List of action names
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot frame
    plot_frame(frame, ax=axes[0], title=f"Frame (True Action: {true_action})")
    
    # Plot action probabilities
    plot_action_distribution(pred_action_probs, action_names, ax=axes[1], title="Predicted Action Probabilities")
    
    plt.tight_layout()
    

def visualize_batch(frames, true_actions, pred_actions, n_samples=4):
    """
    Visualize a batch of frames with their true and predicted actions.
    
    Args:
        frames (numpy.ndarray): Batch of frames
        true_actions (numpy.ndarray): True actions
        pred_actions (numpy.ndarray): Predicted actions
        n_samples (int): Number of samples to visualize
    """
    n_samples = min(n_samples, len(frames))
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(8, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        # Ensure frame is a numpy array
        if isinstance(frames[i], torch.Tensor):
            frame = frames[i].cpu().numpy()
        else:
            frame = frames[i]
        
        # If frame has a channel dimension and it's 1, squeeze it
        if frame.shape[0] == 1:
            frame = frame.squeeze(0)
        
        # If frame has values in [0, 1], scale to [0, 255]
        if frame.max() <= 1.0:
            frame = frame * 255
        
        # Get true and predicted actions
        true_action = true_actions[i].item() if isinstance(true_actions[i], torch.Tensor) else true_actions[i]
        pred_action = pred_actions[i].item() if isinstance(pred_actions[i], torch.Tensor) else pred_actions[i]
        
        # Plot frame
        axes[i].imshow(frame, cmap='gray')
        axes[i].set_title(f"True Action: {true_action}, Predicted Action: {pred_action}")
        axes[i].axis('off')
    
    plt.tight_layout()


def save_visualizations(out_path, epoch, model, dataloader, device, n_samples=8):
    """
    Save visualizations of the model's predictions.
    
    Args:
        out_path (str): Output path
        epoch (int): Current epoch
        model (nn.Module): JEPA model
        dataloader (DataLoader): DataLoader with samples
        device (torch.device): Device to run inference on
        n_samples (int): Number of samples to visualize
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of samples
    batch = next(iter(dataloader))
    frames = batch['frame'].to(device)
    action_idx = batch['action_idx'].to(device)
    
    # Get predictions
    with torch.no_grad():
        action_probs = model.predict_action(frames)
        pred_actions = torch.argmax(action_probs, dim=1)
    
    # Create visualizations
    plt.figure(figsize=(12, 12))
    visualize_batch(frames[:n_samples], action_idx[:n_samples], pred_actions[:n_samples], n_samples=n_samples)
    
    # Save visualization
    plt.savefig(f"{out_path}/predictions_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
    plt.close()
