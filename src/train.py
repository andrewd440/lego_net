#!/usr/bin/env python3
"""Main training script for 3D shape classification"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_model
from src.data.dataset import create_dataloaders
from src.utils.config import load_config, save_config
from src.utils.metrics import calculate_metrics, plot_confusion_matrix
from src.utils.visualization import plot_training_history


class Trainer:
    """Training manager for 3D shape classification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.start_time = datetime.now()
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self.writer = SummaryWriter(self.log_dir)
        self.use_wandb = config['logging']['use_wandb']
        
        if self.use_wandb:
            wandb.init(
                project=config['logging']['project_name'],
                config=config,
                name=f"run_{self.start_time.strftime('%Y%m%d_%H%M%S')}",
            )
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        device_name = self.config['training']['device']
        
        if device_name == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using Metal Performance Shaders (MPS)")
        elif device_name == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        
        return device
    
    def _setup_directories(self):
        """Create necessary directories"""
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        self.log_dir = Path(self.config['training']['log_dir'])
        self.plot_dir = Path(self.config['visualization']['plot_dir'])
        
        for dir_path in [self.checkpoint_dir, self.log_dir, self.plot_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def train(self):
        """Main training loop"""
        print("="*60)
        print("Starting 3D Shape Classification Training")
        print("="*60)
        
        # Save configuration
        save_config(
            self.config,
            self.checkpoint_dir / f"config_{self.start_time.strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        
        # Create data loaders
        print("\nPreparing data...")
        train_loader, val_loader = create_dataloaders(self.config)
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Create model
        print("\nInitializing model...")
        model = create_model(self.config).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Setup optimizer and scheduler
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        print("\nStarting training...")
        for epoch in range(self.config['training']['num_epochs']):
            # Train
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # Validate
            val_loss, val_acc = self._validate(
                model, val_loader, criterion, epoch
            )
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics
            self._log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            if epoch % self.config['logging']['save_frequency'] == 0 or is_best:
                self._save_checkpoint(
                    model, optimizer, epoch, val_acc, val_loss, is_best
                )
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Training complete
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2%}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60)
        
        # Save final plots
        self._save_final_plots()
        
        # Cleanup
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
    
    def _train_epoch(self, model, loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Add channel dimension if needed
            if inputs.dim() == 4:
                inputs = inputs.unsqueeze(1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
            
            # Log batch metrics
            if batch_idx % self.config['logging']['log_interval'] == 0:
                step = epoch * len(loader) + batch_idx
                self.writer.add_scalar('batch/train_loss', loss.item(), step)
                self.writer.add_scalar('batch/train_acc', 100.*correct/total, step)
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate(self, model, loader, criterion, epoch):
        """Validate the model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]")
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if inputs.dim() == 4:
                    inputs = inputs.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Collect for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.*correct/total:.2f}%"
                })
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        
        # Calculate additional metrics
        metrics = calculate_metrics(
            np.array(all_predictions),
            np.array(all_targets)
        )
        
        # Log validation metrics
        self.writer.add_scalar('epoch/val_precision', metrics['precision'], epoch)
        self.writer.add_scalar('epoch/val_recall', metrics['recall'], epoch)
        self.writer.add_scalar('epoch/val_f1', metrics['f1_score'], epoch)
        
        return epoch_loss, epoch_acc
    
    def _create_optimizer(self, model):
        """Create optimizer"""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas'],
            )
        elif opt_config['type'] == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=opt_config['weight_decay'],
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")
        
        return optimizer
    
    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler"""
        sched_config = self.config['scheduler']
        
        if sched_config['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=sched_config['min_lr'],
            )
        elif sched_config['type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Log metrics to tensorboard and wandb"""
        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rates'].append(lr)
        
        # TensorBoard
        self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
        self.writer.add_scalar('epoch/train_acc', train_acc, epoch)
        self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
        self.writer.add_scalar('epoch/val_acc', val_acc, epoch)
        self.writer.add_scalar('epoch/learning_rate', lr, epoch)
        
        # Weights & Biases
        if self.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': lr,
            })
        
        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {lr:.6f}")
    
    def _save_checkpoint(self, model, optimizer, epoch, val_acc, val_loss, is_best):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'config': self.config,
            'history': self.history,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  → Saved best model (val_acc: {val_acc:.2f}%)")
    
    def _save_final_plots(self):
        """Save final training plots"""
        # Training history
        fig, axes = plot_training_history(
            self.history,
            save_path=self.plot_dir / "training_history.png",
            show=False
        )
        
        # Save history as JSON
        with open(self.plot_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train 3D Shape Classification Model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default=None,
        help='Override device from config'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config['training']['device'] = args.device
    
    # Set random seeds for reproducibility
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 