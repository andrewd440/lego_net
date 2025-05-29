"""Test that transforms are being applied correctly"""

import torch
import numpy as np
from src.data.dataset import ModelNet10Voxels
from src.utils.config import load_config

def test_transforms():
    """Test if transforms are being applied"""
    
    # Load config
    config = load_config('configs/default.yaml')
    
    # Create dataset with transforms
    print("Creating dataset with transforms...")
    dataset = ModelNet10Voxels(
        root_dir=config['data']['root_dir'],
        split='train',
        voxel_size=config['data']['voxel_size'],
        config=config
    )
    
    print(f"Transform object: {dataset.transform}")
    
    # Get the same sample multiple times to see if it changes
    sample_idx = 0
    original_voxel = dataset.voxels[sample_idx].copy()
    
    samples = []
    for i in range(5):
        voxel, label = dataset[sample_idx]
        samples.append(voxel.numpy())
        print(f"Sample {i}: min={voxel.min():.3f}, max={voxel.max():.3f}, mean={voxel.mean():.3f}")
    
    # Check if samples are different
    all_same = True
    for i in range(1, len(samples)):
        if not np.allclose(samples[0], samples[i]):
            all_same = False
            break
    
    if all_same:
        print("\n❌ PROBLEM: All samples are identical - transforms not being applied!")
    else:
        print("\n✅ SUCCESS: Samples are different - transforms are being applied!")
    
    # Check what transforms are in the pipeline
    if dataset.transform is not None:
        print(f"\nTransforms in pipeline:")
        if hasattr(dataset.transform, 'transforms'):
            for t in dataset.transform.transforms:
                print(f"  - {t.__class__.__name__}")
        else:
            print(f"  - {dataset.transform.__class__.__name__}")
    else:
        print("\n❌ No transforms found!")

if __name__ == '__main__':
    test_transforms() 