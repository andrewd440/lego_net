#!/usr/bin/env python3
"""Retrain model with proper augmentation to fix overfitting"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.voxel_cnn import VoxelCNN
from src.data.dataset import ModelNet10Voxels
from src.utils.config import load_config
from src.data.transforms_simple import get_simple_augmentations


def apply_augmentation(voxel, augmentations):
    """Apply augmentations to a voxel tensor"""
    for aug in augmentations:
        voxel = aug(voxel)
    return voxel


def train_epoch(model, loader, optimizer, criterion, device, augmentations=None):
    """Train for one epoch with augmentation"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for voxels, labels in tqdm(loader, desc="Training"):
        voxels = voxels.to(device)
        labels = labels.to(device)
        
        # Apply augmentations
        if augmentations:
            augmented_voxels = []
            for i in range(voxels.size(0)):
                aug_voxel = apply_augmentation(voxels[i], augmentations)
                augmented_voxels.append(aug_voxel)
            voxels = torch.stack(augmented_voxels)
        
        # Add channel dimension
        if voxels.dim() == 4:
            voxels = voxels.unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(voxels)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for voxels, labels in tqdm(loader, desc="Validating"):
            voxels = voxels.to(device)
            labels = labels.to(device)
            
            if voxels.dim() == 4:
                voxels = voxels.unsqueeze(1)
            
            outputs = model(voxels)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    
    # Load config
    config = load_config('configs/default.yaml')
    
    # Override some settings for better generalization
    config['model']['dropout_rate'] = 0.5  # Keep high dropout
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.lr
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets WITHOUT transforms (we'll apply them manually)
    train_dataset = ModelNet10Voxels(
        root_dir=config['data']['root_dir'],
        split='train',
        voxel_size=config['data']['voxel_size'],
        config=None  # Don't use config transforms
    )
    
    test_dataset = ModelNet10Voxels(
        root_dir=config['data']['root_dir'],
        split='test',
        voxel_size=config['data']['voxel_size'],
        config=None
    )
    
    # Split train into train/val
    val_split = 0.2
    n_val = int(len(train_dataset) * val_split)
    n_train = len(train_dataset) - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create augmentations that work on MPS
    augmentations = get_simple_augmentations()
    
    # Create model
    model = VoxelCNN(config).to(device)
    
    # Use AdamW with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, augmentations
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Test (for monitoring)
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'test_acc': test_acc,
                'config': config
            }, checkpoint_dir / 'best_model.pt')
            
            # Also save epoch checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'test_acc': test_acc,
                'config': config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
            
            print(f"✓ Saved new best model with val_acc: {val_acc:.2f}%, test_acc: {test_acc:.2f}%")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main() 