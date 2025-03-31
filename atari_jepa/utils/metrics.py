"""
Evaluation metrics for the Atari JEPA model.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_accuracy(predictions, targets):
    """
    Compute accuracy of predictions.
    
    Args:
        predictions (torch.Tensor): Predicted action indices
        targets (torch.Tensor): Target action indices
        
    Returns:
        float: Accuracy score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return accuracy_score(targets, predictions)


def compute_top_k_accuracy(action_probs, targets, k=3):
    """
    Compute top-k accuracy of predictions.
    
    Args:
        action_probs (torch.Tensor): Predicted action probabilities
        targets (torch.Tensor): Target action indices
        k (int): Number of top predictions to consider
        
    Returns:
        float: Top-k accuracy score
    """
    if isinstance(action_probs, np.ndarray):
        action_probs = torch.from_numpy(action_probs)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    # Move tensors to CPU if they're on GPU
    action_probs = action_probs.cpu()
    targets = targets.cpu()
    
    # Get top-k predictions
    _, top_k_preds = torch.topk(action_probs, k, dim=1)
    
    # Check if target is in top-k predictions
    targets_expanded = targets.view(-1, 1).expand(-1, k)
    correct = torch.eq(top_k_preds, targets_expanded).any(dim=1)
    
    # Compute accuracy
    top_k_acc = correct.float().mean().item()
    
    return top_k_acc


def compute_precision_recall_f1(predictions, targets, average='macro'):
    """
    Compute precision, recall, and F1 score.
    
    Args:
        predictions (torch.Tensor): Predicted action indices
        targets (torch.Tensor): Target action indices
        average (str): Averaging method for multi-class metrics
        
    Returns:
        tuple: Precision, recall, and F1 scores
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    precision = precision_score(targets, predictions, average=average, zero_division=0)
    recall = recall_score(targets, predictions, average=average, zero_division=0)
    f1 = f1_score(targets, predictions, average=average, zero_division=0)
    
    return precision, recall, f1


def compute_confusion_matrix(predictions, targets, normalize=None):
    """
    Compute confusion matrix.
    
    Args:
        predictions (torch.Tensor): Predicted action indices
        targets (torch.Tensor): Target action indices
        normalize (str, optional): Normalization method
        
    Returns:
        numpy.ndarray: Confusion matrix
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return confusion_matrix(targets, predictions, normalize=normalize)


def compute_embedding_similarity(embeddings1, embeddings2):
    """
    Compute cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1 (torch.Tensor): First set of embeddings
        embeddings2 (torch.Tensor): Second set of embeddings
        
    Returns:
        torch.Tensor: Cosine similarity matrix
    """
    # Normalize embeddings
    embeddings1 = F.normalize(embeddings1, dim=1)
    embeddings2 = F.normalize(embeddings2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.mm(embeddings1, embeddings2.t())
    
    return similarity


def compute_embedding_retrieval_metrics(predicted_embeddings, target_embeddings, k_values=[1, 5, 10]):
    """
    Compute retrieval metrics for embeddings.
    
    Args:
        predicted_embeddings (torch.Tensor): Predicted embeddings
        target_embeddings (torch.Tensor): Target embeddings
        k_values (list): List of k values for Recall@k
        
    Returns:
        dict: Dictionary of retrieval metrics
    """
    # Compute similarity matrix
    similarity = compute_embedding_similarity(predicted_embeddings, target_embeddings)
    
    # Get indices of targets (ground truth is the diagonal)
    batch_size = similarity.size(0)
    targets = torch.arange(batch_size, device=similarity.device)
    
    # Sort similarity scores
    _, indices = similarity.sort(descending=True, dim=1)
    
    # Compute rank of correct target for each prediction
    correct_indices = (indices == targets.view(-1, 1)).nonzero()
    ranks = correct_indices[:, 1] + 1  # +1 because indices are 0-based
    
    # Compute Mean Reciprocal Rank (MRR)
    mrr = (1.0 / ranks.float()).mean().item()
    
    # Compute Recall@k for different values of k
    recall_at_k = {}
    for k in k_values:
        recall = (ranks <= k).float().mean().item()
        recall_at_k[f'R@{k}'] = recall
    
    # Compute Median Rank
    median_rank = torch.median(ranks.float()).item()
    
    # Combine metrics
    metrics = {
        'MRR': mrr,
        'MedianRank': median_rank,
        **recall_at_k
    }
    
    return metrics


def evaluate_jepa_model(model, dataloader, device):
    """
    Evaluate JEPA model on a dataset.
    
    Args:
        model (nn.Module): JEPA model
        dataloader (DataLoader): DataLoader with evaluation data
        device (torch.device): Device to run evaluation on
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_pred_actions = []
    all_true_actions = []
    all_context_embeddings = []
    all_predicted_embeddings = []
    all_target_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            frames = batch['frame'].to(device)
            action_idx = batch['action_idx'].to(device)
            actions = batch['action'].to(device)
            
            # Forward pass
            outputs = model(frames, actions, action_idx)
            
            # Get action predictions
            action_probs = model.predict_action(frames)
            pred_actions = torch.argmax(action_probs, dim=1)
            
            # Store predictions and targets
            all_pred_actions.append(pred_actions.cpu())
            all_true_actions.append(action_idx.cpu())
            
            # Store embeddings
            all_context_embeddings.append(outputs['context_embeddings'].cpu())
            all_predicted_embeddings.append(outputs['predicted_target_embeddings'].cpu())
            all_target_embeddings.append(outputs['target_embeddings'].cpu())
    
    # Concatenate all predictions and targets
    all_pred_actions = torch.cat(all_pred_actions)
    all_true_actions = torch.cat(all_true_actions)
    all_context_embeddings = torch.cat(all_context_embeddings)
    all_predicted_embeddings = torch.cat(all_predicted_embeddings)
    all_target_embeddings = torch.cat(all_target_embeddings)
    
    # Compute action prediction metrics
    accuracy = compute_accuracy(all_pred_actions, all_true_actions)
    precision, recall, f1 = compute_precision_recall_f1(all_pred_actions, all_true_actions)
    
    # Compute embedding retrieval metrics
    retrieval_metrics = compute_embedding_retrieval_metrics(
        all_predicted_embeddings, all_target_embeddings
    )
    
    # Combine metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        **retrieval_metrics
    }
    
    return metrics
