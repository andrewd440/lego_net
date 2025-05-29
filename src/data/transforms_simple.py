"""Simplified 3D data augmentation transforms for MPS compatibility"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
import torch.nn.functional as F


class SimpleRotation3D(nn.Module):
    """Simple 90-degree rotation augmentation for MPS compatibility."""
    
    def __init__(self, p: float = 0.5):
        """
        Initialize rotation.
        
        Args:
            p: Probability of applying rotation
        """
        super().__init__()
        self.p = p
    
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """Apply 90-degree rotations to voxel grid."""
        if torch.rand(1).item() > self.p:
            return voxels
        
        # Choose random number of 90-degree rotations
        k = torch.randint(1, 4, (1,)).item()
        
        # Choose random axis
        axis = torch.randint(0, 3, (1,)).item()
        
        # Apply rotation using transpose and flip operations
        if axis == 0:  # Rotate around X axis
            for _ in range(k):
                voxels = voxels.transpose(-2, -1).flip(-1)
        elif axis == 1:  # Rotate around Y axis
            for _ in range(k):
                voxels = voxels.transpose(-3, -1).flip(-1)
        else:  # Rotate around Z axis
            for _ in range(k):
                voxels = voxels.transpose(-3, -2).flip(-2)
        
        return voxels


class SimpleFlip3D(nn.Module):
    """Random flipping augmentation."""
    
    def __init__(self, p: float = 0.5):
        """
        Initialize flipping.
        
        Args:
            p: Probability of flipping per axis
        """
        super().__init__()
        self.p = p
    
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """Apply random flips to voxel grid."""
        # Flip along each axis with probability p
        for dim in [-3, -2, -1]:
            if torch.rand(1).item() < self.p:
                voxels = voxels.flip(dim)
        
        return voxels


class SimpleNoise3D(nn.Module):
    """Add simple noise to voxel grid."""
    
    def __init__(self, noise_level: float = 0.1, p: float = 0.5):
        """
        Initialize noise.
        
        Args:
            noise_level: Amount of noise to add
            p: Probability of applying noise
        """
        super().__init__()
        self.noise_level = noise_level
        self.p = p
    
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """Add noise to voxels."""
        if torch.rand(1).item() > self.p:
            return voxels
        
        # Add uniform noise
        noise = (torch.rand_like(voxels) - 0.5) * self.noise_level
        voxels = voxels + noise
        
        # Clamp to valid range
        voxels = torch.clamp(voxels, 0, 1)
        
        return voxels


class SimpleDropout3D(nn.Module):
    """Randomly drop voxels (set to 0)."""
    
    def __init__(self, drop_rate: float = 0.1, p: float = 0.5):
        """
        Initialize dropout.
        
        Args:
            drop_rate: Fraction of voxels to drop
            p: Probability of applying dropout
        """
        super().__init__()
        self.drop_rate = drop_rate
        self.p = p
    
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """Apply dropout to voxels."""
        if torch.rand(1).item() > self.p:
            return voxels
        
        # Create dropout mask
        mask = torch.rand_like(voxels) > self.drop_rate
        voxels = voxels * mask.float()
        
        return voxels


class SimpleCutout3D(nn.Module):
    """Cut out random cubic regions."""
    
    def __init__(self, size: int = 8, n_holes: int = 1, p: float = 0.5):
        """
        Initialize cutout.
        
        Args:
            size: Size of cubic holes
            n_holes: Number of holes
            p: Probability of applying cutout
        """
        super().__init__()
        self.size = size
        self.n_holes = n_holes
        self.p = p
    
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """Apply cutout to voxels."""
        if torch.rand(1).item() > self.p:
            return voxels
        
        d, h, w = voxels.shape[-3:]
        
        for _ in range(self.n_holes):
            # Random center
            cx = torch.randint(0, d, (1,)).item()
            cy = torch.randint(0, h, (1,)).item()
            cz = torch.randint(0, w, (1,)).item()
            
            # Calculate bounds
            x1 = max(0, cx - self.size // 2)
            x2 = min(d, cx + self.size // 2)
            y1 = max(0, cy - self.size // 2)
            y2 = min(h, cy + self.size // 2)
            z1 = max(0, cz - self.size // 2)
            z2 = min(w, cz + self.size // 2)
            
            # Cut out region
            voxels[..., x1:x2, y1:y2, z1:z2] = 0
        
        return voxels


def get_simple_augmentations():
    """Get list of simple augmentations that work on MPS."""
    return [
        SimpleRotation3D(p=0.5),
        SimpleFlip3D(p=0.3),
        SimpleNoise3D(noise_level=0.05, p=0.5),
        SimpleDropout3D(drop_rate=0.05, p=0.3),
        SimpleCutout3D(size=6, n_holes=1, p=0.3)
    ] 