"""3D data augmentation transforms"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import torch.nn.functional as F


class RandomRotation3D(nn.Module):
    """Random 3D rotation augmentation."""
    
    def __init__(self, degrees: float = 180.0, axis: Optional[List[str]] = None):
        """
        Initialize random rotation.
        
        Args:
            degrees: Maximum rotation angle in degrees
            axis: List of axes to rotate around ['x', 'y', 'z']
        """
        super().__init__()
        self.degrees = degrees
        self.axis = axis or ['z']  # Default to Z-axis rotation only
    
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """Apply random rotation to voxel grid."""
        
        # Random angle
        angle = torch.rand(1).item() * 2 * self.degrees - self.degrees
        angle_rad = np.radians(angle)
        
        # Choose random axis
        axis = np.random.choice(self.axis)
        
        # Create rotation matrix
        if axis == 'x':
            rotation_matrix = torch.tensor([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ], dtype=torch.float32)
        elif axis == 'y':
            rotation_matrix = torch.tensor([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ], dtype=torch.float32)
        else:  # z
            rotation_matrix = torch.tensor([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ], dtype=torch.float32)
        
        # Apply rotation using grid sampling
        return self._rotate_voxels(voxels, rotation_matrix)
    
    def _rotate_voxels(self, voxels: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Apply rotation matrix to voxel grid using grid sampling."""
        # Add batch and channel dimensions if needed
        squeeze_dims = []
        if voxels.dim() == 3:
            voxels = voxels.unsqueeze(0).unsqueeze(0)
            squeeze_dims = [0, 1]
        elif voxels.dim() == 4:
            voxels = voxels.unsqueeze(0)
            squeeze_dims = [0]
        
        # Create coordinate grid
        d, h, w = voxels.shape[-3:]
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, d),
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij'
        ), dim=-1)
        
        # Flatten and apply rotation
        grid_flat = grid.reshape(-1, 3)
        rotated_grid = torch.matmul(grid_flat, rotation_matrix.T)
        rotated_grid = rotated_grid.reshape(d, h, w, 3)
        
        # Grid sample expects (N, D, H, W, 3) format
        rotated_grid = rotated_grid.unsqueeze(0)
        
        # Apply grid sampling
        rotated = F.grid_sample(
            voxels,
            rotated_grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True
        )
        
        # Remove added dimensions
        for dim in reversed(squeeze_dims):
            rotated = rotated.squeeze(dim)
        
        return rotated


class RandomScale3D(nn.Module):
    """Random 3D scaling augmentation."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        """
        Initialize random scaling.
        
        Args:
            scale_range: Min and max scale factors
        """
        super().__init__()
        self.scale_range = scale_range
    
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """Apply random scaling to voxel grid."""
        
        # Random scale factor
        scale = torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        
        # Apply scaling using interpolation
        if voxels.dim() == 3:
            voxels = voxels.unsqueeze(0).unsqueeze(0)
            scaled = F.interpolate(voxels, scale_factor=scale, mode='nearest')
            
            # Crop or pad to original size
            target_size = voxels.shape[-3:]
            scaled = self._resize_to_target(scaled, target_size)
            scaled = scaled.squeeze(0).squeeze(0)
        else:
            scaled = F.interpolate(voxels, scale_factor=scale, mode='nearest')
            target_size = voxels.shape[-3:]
            scaled = self._resize_to_target(scaled, target_size)
        
        return scaled
    
    def _resize_to_target(self, tensor: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
        """Resize tensor to target size by cropping or padding."""
        current_size = tensor.shape[-3:]
        
        # Calculate padding/cropping for each dimension
        pad_crop = []
        for curr, tgt in zip(current_size, target_size):
            if curr < tgt:
                # Need padding
                pad_total = tgt - curr
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                pad_crop.extend([pad_left, pad_right])
            else:
                # Need cropping
                pad_crop.extend([0, 0])
        
        # Apply padding if needed
        if any(p > 0 for p in pad_crop):
            tensor = F.pad(tensor, pad_crop[::-1])  # Reverse for correct order
        
        # Apply cropping if needed
        for i, (curr, tgt) in enumerate(zip(tensor.shape[-3:], target_size)):
            if curr > tgt:
                crop_total = curr - tgt
                crop_start = crop_total // 2
                if i == 0:
                    tensor = tensor[..., crop_start:crop_start+tgt, :, :]
                elif i == 1:
                    tensor = tensor[..., :, crop_start:crop_start+tgt, :]
                else:
                    tensor = tensor[..., :, :, crop_start:crop_start+tgt]
        
        return tensor


class RandomNoise3D(nn.Module):
    """Add random noise to voxel grid."""
    
    def __init__(self, noise_std: float = 0.01, noise_prob: float = 0.1):
        """
        Initialize random noise.
        
        Args:
            noise_std: Standard deviation of Gaussian noise
            noise_prob: Probability of flipping voxel values
        """
        super().__init__()
        self.noise_std = noise_std
        self.noise_prob = noise_prob
    
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """Add random noise to voxels."""
        
        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(voxels) * self.noise_std
            voxels = voxels + noise
        
        # Random flipping
        if self.noise_prob > 0:
            flip_mask = torch.rand_like(voxels) < self.noise_prob
            voxels = torch.where(flip_mask, 1 - voxels, voxels)
        
        # Clamp to valid range
        voxels = torch.clamp(voxels, 0, 1)
        
        return voxels


class Compose3D(nn.Module):
    """Compose multiple 3D transforms."""
    
    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
    
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            voxels = transform(voxels)
        return voxels


def get_transforms(split: str, config: Optional[Dict[str, Any]] = None) -> Optional[nn.Module]:
    """
    Get transforms for a given split.
    
    Args:
        split: 'train' or 'test'
        config: Configuration dictionary
        
    Returns:
        Transform module or None
    """
    if split != 'train' or config is None:
        return None
    
    aug_config = config['data'].get('augmentation', {})
    
    transforms = []
    
    # Skip rotation for MPS compatibility (grid_sample not implemented)
    # if aug_config.get('random_rotation', False):
    #     transforms.append(RandomRotation3D(degrees=180, axis=['z']))
    
    if aug_config.get('random_scale', False):
        scale_range = aug_config.get('scale_range', [0.8, 1.2])
        transforms.append(RandomScale3D(scale_range=tuple(scale_range)))
    
    if aug_config.get('random_noise', False):
        noise_std = aug_config.get('noise_std', 0.01)
        transforms.append(RandomNoise3D(noise_std=noise_std))
    
    if transforms:
        return Compose3D(transforms)
    
    return None 