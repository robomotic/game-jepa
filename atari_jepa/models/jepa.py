"""
JEPA (Joint Embedding Predictive Architecture) model.
Combines context encoder, target encoder, and predictor for frame-to-action prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .context_encoder import ContextEncoder, ResNetContextEncoder, VisionTransformerEncoder
from .target_encoder import TargetEncoder, AtariActionEncoder
from .predictor import Predictor, ResidualPredictor, TransformerPredictor


class JEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture (JEPA) for Atari frame-to-action prediction.
    """
    
    def __init__(
        self,
        input_channels=1,
        action_dim=18,
        embedding_dim=256,
        hidden_dim=512,
        context_encoder_type="standard",
        target_encoder_type="standard",
        predictor_type="standard",
        temperature=0.07,
        use_masking=False,
        mask_ratio=0.4,
    ):
        """
        Initialize the JEPA model.
        
        Args:
            input_channels (int): Number of input channels for the context encoder
            action_dim (int): Dimension of the action space
            embedding_dim (int): Dimension of the embeddings
            hidden_dim (int): Dimension of the hidden layers
            context_encoder_type (str): Type of context encoder ('standard' or 'resnet')
            target_encoder_type (str): Type of target encoder ('standard' or 'embedding')
            predictor_type (str): Type of predictor ('standard' or 'residual')
            temperature (float): Temperature for contrastive loss
        """
        super(JEPA, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.use_masking = use_masking
        self.mask_ratio = mask_ratio
        
        # Context encoder (frame encoder)
        if context_encoder_type == "standard":
            self.context_encoder = ContextEncoder(
                input_channels=input_channels,
                embedding_dim=embedding_dim
            )
        elif context_encoder_type == "resnet":
            self.context_encoder = ResNetContextEncoder(
                input_channels=input_channels,
                embedding_dim=embedding_dim
            )
        elif context_encoder_type == "vit":
            self.context_encoder = VisionTransformerEncoder(
                img_size=84,  # Standard Atari frame size
                patch_size=7,  # 7x7 patches
                in_chans=input_channels,
                embed_dim=embedding_dim,
                depth=8,       # Number of transformer blocks
                num_heads=8,   # Number of attention heads
                mlp_ratio=4.
            )
        else:
            raise ValueError(f"Unknown context encoder type: {context_encoder_type}")
        
        # Target encoder (action encoder)
        if target_encoder_type == "standard":
            self.target_encoder = TargetEncoder(
                action_dim=action_dim,
                embedding_dim=embedding_dim
            )
        elif target_encoder_type == "embedding":
            self.target_encoder = AtariActionEncoder(
                action_dim=action_dim,
                embedding_dim=embedding_dim
            )
        else:
            raise ValueError(f"Unknown target encoder type: {target_encoder_type}")
        
        # Predictor
        if predictor_type == "standard" or predictor_type == "mlp":
            self.predictor = Predictor(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim
            )
        elif predictor_type == "residual":
            self.predictor = ResidualPredictor(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim
            )
        elif predictor_type == "transformer":
            self.predictor = TransformerPredictor(
                embed_dim=embedding_dim,
                output_dim=embedding_dim,
                num_heads=8,
                num_layers=4,
                dropout=0.1
            )
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
    
    def encode_context(self, x, apply_mask=False):
        """
        Encode context (frames) using the context encoder.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            apply_mask (bool): Whether to apply masking for self-supervised learning
            
        Returns:
            Tensor: Encoded context embeddings
        """
        if self.use_masking and apply_mask:
            # Apply random masking for self-supervised learning
            # For Vision Transformer, this would be patch masking
            if isinstance(self.context_encoder, VisionTransformerEncoder):
                # ViT masking happens in the encoder itself
                return self.context_encoder(x, mask_ratio=self.mask_ratio)
            else:
                # For CNN/ResNet, we can apply simple random masking to the input
                batch_size, channels, height, width = x.shape
                mask = torch.ones_like(x)
                
                # Create random mask (1=keep, 0=mask)
                for i in range(batch_size):
                    num_masks = int(height * width * self.mask_ratio)
                    mask_indices = torch.randperm(height * width)[:num_masks]
                    mask_y = mask_indices // width
                    mask_x = mask_indices % width
                    mask[i, :, mask_y, mask_x] = 0.0
                
                # Apply mask and encode
                masked_x = x * mask
                return self.context_encoder(masked_x)
        
        # No masking, standard encoding
        return self.context_encoder(x)
    
    def encode_target(self, x):
        """
        Encode target (actions) using the target encoder.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, action_dim) or (batch_size,)
            
        Returns:
            Tensor: Encoded target embeddings
        """
        return self.target_encoder(x)
    
    def predict(self, context_embeddings):
        """
        Predict target embeddings from context embeddings.
        
        Args:
            context_embeddings (Tensor): Context embeddings
            
        Returns:
            Tensor: Predicted target embeddings
        """
        return self.predictor(context_embeddings)
    
    def forward(self, frames, actions=None, action_idx=None, apply_mask=True):
        """
        Forward pass of the JEPA model.
        
        Args:
            frames (Tensor): Input frames
            actions (Tensor, optional): One-hot encoded actions
            action_idx (Tensor, optional): Action indices
            apply_mask (bool): Whether to apply masking for self-supervised learning
            
        Returns:
            dict: Dictionary containing model outputs
        """
        # Encode context (frames) with optional masking
        context_embeddings = self.encode_context(frames, apply_mask=apply_mask)
        
        # Predict target embeddings
        predicted_target_embeddings = self.predict(context_embeddings)
        
        outputs = {
            'context_embeddings': context_embeddings,
            'predicted_target_embeddings': predicted_target_embeddings
        }
        
        # Choose which action representation to use based on the target encoder type
        if isinstance(self.target_encoder, AtariActionEncoder):
            # AtariActionEncoder expects action indices
            if action_idx is not None:
                target_embeddings = self.encode_target(action_idx)
                outputs['target_embeddings'] = target_embeddings
            elif actions is not None:
                # If only one-hot actions are provided, use the forward_one_hot method
                target_embeddings = self.target_encoder.forward_one_hot(actions)
                outputs['target_embeddings'] = target_embeddings
        else:
            # Standard TargetEncoder expects one-hot actions
            if actions is not None:
                target_embeddings = self.encode_target(actions)
                outputs['target_embeddings'] = target_embeddings
            elif action_idx is not None:
                # Convert indices to one-hot if needed
                batch_size = action_idx.shape[0]
                action_dim = self.target_encoder.action_dim
                actions = torch.zeros(batch_size, action_dim, device=action_idx.device)
                actions.scatter_(1, action_idx.unsqueeze(1), 1.0)
                target_embeddings = self.encode_target(actions)
                outputs['target_embeddings'] = target_embeddings
        
        return outputs
    
    def compute_similarity(self, predicted_embeddings, target_embeddings):
        """
        Compute similarity between predicted and target embeddings.
        
        Args:
            predicted_embeddings (Tensor): Predicted embeddings
            target_embeddings (Tensor): Target embeddings
            
        Returns:
            Tensor: Similarity matrix
        """
        # Normalize embeddings
        predicted_embeddings = F.normalize(predicted_embeddings, dim=1)
        target_embeddings = F.normalize(target_embeddings, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(predicted_embeddings, target_embeddings.T) / self.temperature
        
        return similarity
    
    def compute_loss(self, outputs):
        """
        Compute the contrastive loss for JEPA.
        
        Args:
            outputs (dict): Model outputs from forward pass
            
        Returns:
            Tensor: Loss value
        """
        predicted_embeddings = outputs['predicted_target_embeddings']
        target_embeddings = outputs['target_embeddings']
        
        # Compute similarity matrix
        similarity = self.compute_similarity(predicted_embeddings, target_embeddings)
        
        # Define positive pairs (diagonal elements)
        batch_size = similarity.size(0)
        labels = torch.arange(batch_size, device=similarity.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss
    
    def predict_action(self, frames, action_space_size=18):
        """
        Predict actions from frames.
        
        Args:
            frames (Tensor): Input frames
            action_space_size (int): Size of the action space
            
        Returns:
            Tensor: Predicted action probabilities
        """
        # Encode frames
        context_embeddings = self.encode_context(frames)
        
        # Predict target embeddings
        predicted_target_embeddings = self.predict(context_embeddings)
        
        # Normalize predicted embeddings
        predicted_target_embeddings = F.normalize(predicted_target_embeddings, dim=1)
        
        # Generate action embeddings for all possible actions
        action_embeddings = []
        for i in range(action_space_size):
            # Create a batch of the same action
            action_idx = torch.full((frames.size(0),), i, device=frames.device, dtype=torch.long)
            action_embedding = self.encode_target(action_idx)
            action_embedding = F.normalize(action_embedding, dim=1)
            action_embeddings.append(action_embedding)
        
        # Stack action embeddings
        action_embeddings = torch.stack(action_embeddings, dim=1)  # [batch_size, n_actions, embedding_dim]
        
        # Compute similarity for each action
        predicted_target_embeddings = predicted_target_embeddings.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        similarities = torch.bmm(predicted_target_embeddings, action_embeddings.transpose(1, 2))  # [batch_size, 1, n_actions]
        similarities = similarities.squeeze(1)  # [batch_size, n_actions]
        
        # Apply softmax to get probabilities
        action_probs = F.softmax(similarities / self.temperature, dim=1)
        
        return action_probs
