"""Monitor training progress and check for improvements"""

import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def monitor_latest_checkpoint():
    """Check the latest checkpoint for training progress"""
    checkpoint_dir = Path('checkpoints')
    
    # Find the latest checkpoint
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        print("No checkpoints found yet...")
        return None
    
    latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    
    # Check if there's a training history plot
    plot_dir = Path('plots')
    history_file = plot_dir / 'training_history.json'
    
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        print(f"\nLatest Training Progress:")
        print(f"Epochs completed: {len(history['train_acc'])}")
        
        if history['train_acc']:
            print(f"Latest Train Acc: {history['train_acc'][-1]:.2f}%")
            print(f"Latest Val Acc: {history['val_acc'][-1]:.2f}%")
            print(f"Best Val Acc: {max(history['val_acc']):.2f}%")
            
            # Check if overfitting is reduced
            train_val_gap = history['train_acc'][-1] - history['val_acc'][-1]
            print(f"Train-Val Gap: {train_val_gap:.2f}%")
            
            if train_val_gap < 10:
                print("✅ Good generalization - gap < 10%")
            else:
                print("⚠️  Potential overfitting - gap > 10%")
    
    return latest

def check_tensorboard_logs():
    """Check if TensorBoard logs are being generated"""
    log_dir = Path('logs')
    if log_dir.exists():
        events = list(log_dir.rglob('events.out.tfevents.*'))
        if events:
            print(f"\n✅ TensorBoard logs found: {len(events)} event files")
            print("Run 'tensorboard --logdir logs' to visualize")
        else:
            print("\n❌ No TensorBoard logs found")

def estimate_time_remaining():
    """Estimate remaining training time based on checkpoint intervals"""
    checkpoint_dir = Path('checkpoints')
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'), 
                        key=lambda p: p.stat().st_mtime)
    
    if len(checkpoints) >= 2:
        # Calculate time between checkpoints
        time_diff = checkpoints[-1].stat().st_mtime - checkpoints[-2].stat().st_mtime
        epoch1 = int(checkpoints[-2].stem.split('_')[-1])
        epoch2 = int(checkpoints[-1].stem.split('_')[-1])
        epochs_diff = epoch2 - epoch1
        
        if epochs_diff > 0:
            time_per_epoch = time_diff / epochs_diff
            remaining_epochs = 100 - epoch2  # Assuming 100 total epochs
            remaining_time = remaining_epochs * time_per_epoch
            
            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            
            print(f"\n⏱️  Estimated time remaining: {hours}h {minutes}m")
            print(f"   (Based on {time_per_epoch/60:.1f} min/epoch)")

if __name__ == '__main__':
    print("Monitoring training progress...")
    print("=" * 50)
    
    checkpoint = monitor_latest_checkpoint()
    check_tensorboard_logs()
    estimate_time_remaining()
    
    if checkpoint:
        print(f"\n📁 Latest checkpoint: {checkpoint.name}")
    else:
        print("\n⏳ Waiting for first checkpoint...") 