"""Custom loss functions for 3D shape classification"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights (scalar or tensor)
            gamma: Focusing parameter
            reduction: 'none', 'mean', or 'sum'
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Focal loss value
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            targets = F.one_hot(targets, num_classes).float()
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        p = F.softmax(inputs, dim=-1)
        
        # Get class probabilities
        if self.label_smoothing > 0:
            p_t = (p * targets).sum(dim=-1)
        else:
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, float):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_term = alpha_t * focal_term
        
        # Compute final loss
        loss = focal_term * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for voxel-based segmentation tasks.
    Useful for future LEGO brick segmentation.
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            inputs: Predicted probabilities of shape (N, C, D, H, W)
            targets: Ground truth labels of shape (N, D, H, W)
            
        Returns:
            Dice loss value
        """
        # Convert targets to one-hot encoding
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Apply softmax to predictions
        inputs = F.softmax(inputs, dim=1)
        
        # Flatten spatial dimensions
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # Compute intersection and union
        intersection = (inputs * targets_one_hot).sum(dim=2)
        union = inputs.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Compute loss
        loss = 1.0 - dice
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CenterLoss(nn.Module):
    """
    Center Loss for improved feature clustering.
    Helps learn discriminative features for each class.
    """
    
    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 0.5):
        """
        Initialize Center Loss.
        
        Args:
            num_classes: Number of classes
            feat_dim: Feature dimension
            alpha: Learning rate for center updates
        """
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        
        # Initialize class centers
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute center loss.
        
        Args:
            features: Feature vectors of shape (N, D)
            labels: Ground truth labels of shape (N,)
            
        Returns:
            Center loss value
        """
        batch_size = features.size(0)
        
        # Compute distances to centers
        distances = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distances.addmm_(features, self.centers.t(), beta=1, alpha=-2)
        
        # Select distances to correct centers
        labels = labels.unsqueeze(1).expand(batch_size, self.feat_dim)
        centers_batch = self.centers.gather(0, labels.long())
        
        # Compute loss
        loss = F.mse_loss(features, centers_batch)
        
        return loss


def combined_loss(
    logits: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.01,
    focal_gamma: float = 2.0,
    num_classes: int = 10,
    feat_dim: int = 256,
) -> torch.Tensor:
    """
    Combined loss function using CrossEntropy/Focal Loss and Center Loss.
    
    Args:
        logits: Model predictions
        features: Feature vectors from model
        targets: Ground truth labels
        alpha: Weight for classification loss
        beta: Weight for center loss
        focal_gamma: Gamma parameter for focal loss
        num_classes: Number of classes
        feat_dim: Feature dimension
        
    Returns:
        Combined loss value
    """
    # Classification loss (Focal or CrossEntropy)
    if focal_gamma > 0:
        focal_loss = FocalLoss(gamma=focal_gamma)
        cls_loss = focal_loss(logits, targets)
    else:
        cls_loss = F.cross_entropy(logits, targets)
    
    # Center loss for feature clustering
    center_loss = CenterLoss(num_classes, feat_dim)
    center_loss_val = center_loss(features, targets)
    
    # Combine losses
    total_loss = alpha * cls_loss + beta * center_loss_val
    
    return total_loss 