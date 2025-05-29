import numpy as np
from collections import Counter
import torch
from src.data.dataset import ModelNet10Voxels, create_dataloaders
from src.utils.config import load_config

def analyze_distributions():
    """Analyze train/val/test distributions and look for issues"""
    
    config = load_config('configs/default.yaml')
    
    # Load all datasets
    print("Loading datasets...")
    train_full = ModelNet10Voxels(
        root_dir=config['data']['root_dir'],
        split='train',
        voxel_size=config['data']['voxel_size']
    )
    
    test_dataset = ModelNet10Voxels(
        root_dir=config['data']['root_dir'],
        split='test',
        voxel_size=config['data']['voxel_size']
    )
    
    # Get train/val split
    train_loader, val_loader = create_dataloaders(config)
    
    print(f"\nDataset sizes:")
    print(f"  Full training set: {len(train_full)}")
    print(f"  Train split: {len(train_loader.dataset)}")
    print(f"  Val split: {len(val_loader.dataset)}")
    print(f"  Test set: {len(test_dataset)}")
    
    # Analyze class distributions
    print("\nClass distribution in full training set:")
    train_labels = train_full.labels
    train_counts = Counter(train_labels)
    for i, count in sorted(train_counts.items()):
        print(f"  {train_full.get_class_name(i):12s}: {count:4d} ({count/len(train_full)*100:5.1f}%)")
    
    print("\nClass distribution in test set:")
    test_labels = test_dataset.labels
    test_counts = Counter(test_labels)
    for i, count in sorted(test_counts.items()):
        print(f"  {test_dataset.get_class_name(i):12s}: {count:4d} ({count/len(test_dataset)*100:5.1f}%)")
    
    # Check for data leakage - compare some voxel statistics
    print("\nVoxel statistics comparison:")
    
    # Sample some indices
    n_samples = 100
    train_indices = np.random.choice(len(train_full), n_samples)
    test_indices = np.random.choice(len(test_dataset), n_samples)
    
    train_occupancy = []
    test_occupancy = []
    
    for idx in train_indices:
        voxel, _ = train_full[idx]
        train_occupancy.append((voxel > 0.5).sum().item())
    
    for idx in test_indices:
        voxel, _ = test_dataset[idx]
        test_occupancy.append((voxel > 0.5).sum().item())
    
    print(f"  Train mean occupancy: {np.mean(train_occupancy):.1f} ± {np.std(train_occupancy):.1f}")
    print(f"  Test mean occupancy: {np.mean(test_occupancy):.1f} ± {np.std(test_occupancy):.1f}")
    
    # Check if transforms are being applied
    print(f"\nTransforms applied to train: {train_full.transform is not None}")
    print(f"Transforms applied to test: {test_dataset.transform is not None}")
    
    # Sample a few examples to see the actual voxel values
    print("\nSample voxel value ranges:")
    for i in range(3):
        train_voxel, train_label = train_full[i]
        test_voxel, test_label = test_dataset[i]
        print(f"  Train sample {i} ({train_full.get_class_name(train_label)}): [{train_voxel.min():.3f}, {train_voxel.max():.3f}]")
        print(f"  Test sample {i} ({test_dataset.get_class_name(test_label)}): [{test_voxel.min():.3f}, {test_voxel.max():.3f}]")
    
    # Check config for potential issues
    print(f"\nConfiguration settings:")
    print(f"  Random rotation: {config['data']['augmentation']['random_rotation']}")
    print(f"  Random scale: {config['data']['augmentation']['random_scale']}")
    print(f"  Random noise: {config['data']['augmentation']['random_noise']}")
    print(f"  Dropout rate: {config['model']['dropout_rate']}")

if __name__ == '__main__':
    analyze_distributions() 