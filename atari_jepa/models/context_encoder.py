"""
Context Encoder for JEPA model.
Encodes game frames into a latent representation.
Includes CNN, ResNet, and Vision Transformer implementations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat


class ContextEncoder(nn.Module):
    """
    Context Encoder for the JEPA model.
    Encodes game frames into a latent representation.
    """
    
    def __init__(self, input_channels=1, embedding_dim=256):
        """
        Initialize the context encoder.
        
        Args:
            input_channels (int): Number of input channels (1 for grayscale)
            embedding_dim (int): Dimension of the output embeddings
        """
        super(ContextEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # CNN backbone
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the flattened features
        # For 84x84 input, the output size after convolutions is 7x7x64
        self.flatten_size = 7 * 7 * 64
        
        # FC layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, embedding_dim)
        
        # Normalization layer
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        """
        Forward pass of the context encoder.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: Encoded embeddings of shape (batch_size, embedding_dim)
        """
        # CNN backbone
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Normalize embeddings
        x = self.norm(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for the context encoder.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize the residual block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the convolution
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass of the residual block.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output tensor
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetContextEncoder(nn.Module):
    """
    ResNet-based context encoder for the JEPA model.
    Provides a more advanced encoder architecture for better representations.
    """
    
    def __init__(self, input_channels=1, embedding_dim=256):
        """
        Initialize the ResNet context encoder.
        
        Args:
            input_channels (int): Number of input channels
            embedding_dim (int): Dimension of the output embeddings
        """
        super(ResNetContextEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Initial layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Calculate the size of the flattened features
        # For 84x84 input, the output size after convolutions is 6x6x256
        self.flatten_size = 6 * 6 * 256
        
        # FC layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, embedding_dim)
        
        # Normalization layer
        self.norm = nn.LayerNorm(embedding_dim)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a layer of residual blocks.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_blocks (int): Number of residual blocks
            stride (int): Stride for the first block
            
        Returns:
            nn.Sequential: Layer of residual blocks
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the ResNet context encoder.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: Encoded embeddings of shape (batch_size, embedding_dim)
        """
        # Initial layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (6, 6))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Normalize embeddings
        x = self.norm(x)
        
        return x


# Vision Transformer Components adapted from Facebook Research I-JEPA

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sinusoidal positional embeddings.
    
    Args:
        embed_dim (int): Embedding dimension
        grid_size (int): Grid size
        cls_token (bool): Whether to include class token
        
    Returns:
        numpy.ndarray: Positional embeddings
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generate 2D sinusoidal positional embeddings from grid.
    
    Args:
        embed_dim (int): Embedding dimension
        grid (numpy.ndarray): Grid
        
    Returns:
        numpy.ndarray: Positional embeddings
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sinusoidal positional embeddings from grid.
    
    Args:
        embed_dim (int): Embedding dimension
        pos (numpy.ndarray): Positions
        
    Returns:
        numpy.ndarray: Positional embeddings
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding for Vision Transformer.
    """
    def __init__(self, img_size=84, patch_size=7, in_chans=1, embed_dim=768):
        """
        Initialize patch embedding.
        
        Args:
            img_size (int): Input image size
            patch_size (int): Patch size
            in_chans (int): Number of input channels
            embed_dim (int): Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass of the patch embedding.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: Patch embeddings
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        return x


class Attention(nn.Module):
    """
    Self-attention module for Vision Transformer.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        """
        Initialize attention module.
        
        Args:
            dim (int): Dimension
            num_heads (int): Number of attention heads
            qkv_bias (bool): Whether to use bias in QKV projection
            attn_drop (float): Attention dropout rate
            proj_drop (float): Projection dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Forward pass of the attention module.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output tensor
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLPBlock(nn.Module):
    """
    MLP block for Vision Transformer.
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., act_layer=nn.GELU):
        """
        Initialize MLP block.
        
        Args:
            dim (int): Input dimension
            mlp_ratio (float): MLP ratio
            drop (float): Dropout rate
            act_layer (nn.Module): Activation layer
        """
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass of the MLP block.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block for Vision Transformer.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Initialize transformer block.
        
        Args:
            dim (int): Input dimension
            num_heads (int): Number of attention heads
            mlp_ratio (float): MLP ratio
            qkv_bias (bool): Whether to use bias in QKV projection
            drop (float): Dropout rate
            attn_drop (float): Attention dropout rate
            act_layer (nn.Module): Activation layer
            norm_layer (nn.Module): Normalization layer
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(dim, mlp_ratio=mlp_ratio, drop=drop, act_layer=act_layer)

    def forward(self, x):
        """
        Forward pass of the transformer block.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output tensor
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer for frame encoding.
    Based on Facebook Research's I-JEPA implementation.
    """
    def __init__(self, 
                 img_size=84, 
                 patch_size=7, 
                 in_chans=1, 
                 embed_dim=384, 
                 depth=12,
                 num_heads=6, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 norm_layer=nn.LayerNorm):
        """
        Initialize Vision Transformer encoder.
        
        Args:
            img_size (int): Input image size
            patch_size (int): Patch size
            in_chans (int): Number of input channels
            embed_dim (int): Embedding dimension
            depth (int): Depth of transformer
            num_heads (int): Number of attention heads
            mlp_ratio (float): MLP ratio
            qkv_bias (bool): Whether to use bias in QKV projection
            drop_rate (float): Dropout rate
            attn_drop_rate (float): Attention dropout rate
            norm_layer (nn.Module): Normalization layer
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for _ in range(depth)])
        
        # Output normalization and projection
        self.norm = norm_layer(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """
        Initialize weights.
        
        Args:
            m (nn.Module): Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Forward pass of the Vision Transformer encoder.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: Encoded embeddings of shape (batch_size, embedding_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, C)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Global pooling: average across patches
        x = x.mean(dim=1)  # (B, C)
        
        # Final projection
        x = self.fc(x)
        
        return x
