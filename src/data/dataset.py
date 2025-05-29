"""ModelNet10 dataset with voxelization support"""

import os
import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import urllib.request
import zipfile
import shutil

from .voxelization import mesh_to_voxels, batch_voxelize
from .transforms import get_transforms


class ModelNet10Voxels(Dataset):
    """
    ModelNet10 dataset with automatic voxelization and caching.
    
    Downloads ModelNet10 if not present, voxelizes meshes on first access,
    and caches results for faster subsequent loading.
    """
    
    MODELNET10_URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    CLASSES = [
        'bathtub', 'bed', 'chair', 'desk', 'dresser',
        'monitor', 'night_stand', 'sofa', 'table', 'toilet'
    ]
    
    def __init__(
        self,
        root_dir: str = "./data/ModelNet10",
        split: str = "train",
        voxel_size: int = 32,
        config: Optional[Dict[str, Any]] = None,
        download: bool = True,
        force_rebuild: bool = False,
    ):
        """
        Initialize ModelNet10 dataset.
        
        Args:
            root_dir: Root directory for dataset
            split: 'train' or 'test'
            voxel_size: Size of voxel grid
            config: Configuration dictionary
            download: Whether to download dataset if not present
            force_rebuild: Force rebuild of voxel cache
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.voxel_size = voxel_size
        self.config = config or {}
        
        # Setup paths
        self.raw_dir = self.root_dir / "ModelNet10"
        self.cache_dir = Path(config['data']['cache_dir']) if config else self.root_dir / "cache"
        self.cache_file = self.cache_dir / f"modelnet10_{split}_{voxel_size}.h5"
        
        # Download if needed
        if download and not self.raw_dir.exists():
            self._download()
        
        # Build or load cache
        if force_rebuild or not self.cache_file.exists():
            self._build_cache()
        
        # Load data from cache
        self._load_cache()
        
        # Setup transforms
        self.transform = get_transforms(split, config) if config else None
    
    def _download(self):
        """Download and extract ModelNet10 dataset."""
        print(f"Downloading ModelNet10 dataset to {self.root_dir}...")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        zip_path = self.root_dir / "ModelNet10.zip"
        
        # Download with progress bar
        def download_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded / total_size * 100, 100)
            print(f"\rDownloading: {percent:.1f}%", end='')
        
        urllib.request.urlretrieve(self.MODELNET10_URL, zip_path, reporthook=download_hook)
        print("\nExtracting...")
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)
        
        # Cleanup
        zip_path.unlink()
        print("Download complete!")
    
    def _build_cache(self):
        """Build voxel cache from mesh files."""
        print(f"Building voxel cache for {self.split} split...")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all mesh files
        mesh_files = []
        labels = []
        
        for class_idx, class_name in enumerate(self.CLASSES):
            class_dir = self.raw_dir / class_name / self.split
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found")
                continue
            
            # Find OFF files
            off_files = list(class_dir.glob("*.off"))
            mesh_files.extend(off_files)
            labels.extend([class_idx] * len(off_files))
        
        print(f"Found {len(mesh_files)} mesh files")
        
        # Voxelize in batches
        voxels_list = []
        batch_size = 100
        
        for i in tqdm(range(0, len(mesh_files), batch_size), desc="Voxelizing"):
            batch_files = mesh_files[i:i+batch_size]
            batch_voxels = []
            
            for mesh_file in batch_files:
                try:
                    voxels = mesh_to_voxels(mesh_file, self.voxel_size)
                    batch_voxels.append(voxels.numpy())
                except Exception as e:
                    print(f"\nError processing {mesh_file}: {e}")
                    # Add zero voxels as fallback
                    batch_voxels.append(np.zeros((self.voxel_size,) * 3, dtype=np.float32))
            
            voxels_list.extend(batch_voxels)
        
        # Convert to arrays
        voxels_array = np.stack(voxels_list)
        labels_array = np.array(labels, dtype=np.int64)
        
        # Save to HDF5
        print(f"Saving cache to {self.cache_file}")
        with h5py.File(self.cache_file, 'w') as f:
            f.create_dataset('voxels', data=voxels_array, compression='gzip')
            f.create_dataset('labels', data=labels_array)
            f.attrs['num_classes'] = len(self.CLASSES)
            f.attrs['voxel_size'] = self.voxel_size
            f.attrs['split'] = self.split
        
        print("Cache building complete!")
    
    def _load_cache(self):
        """Load data from cache file."""
        with h5py.File(self.cache_file, 'r') as f:
            self.voxels = f['voxels'][:]
            self.labels = f['labels'][:]
        
        print(f"Loaded {len(self.voxels)} samples from cache")
    
    def __len__(self) -> int:
        return len(self.voxels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            voxels: Voxel grid tensor
            label: Class label
        """
        voxels = torch.from_numpy(self.voxels[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Apply transforms
        if self.transform:
            voxels = self.transform(voxels)
        
        return voxels, label
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index."""
        return self.CLASSES[class_idx]
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a sample."""
        return {
            'index': idx,
            'class_idx': int(self.labels[idx]),
            'class_name': self.get_class_name(self.labels[idx]),
            'voxel_shape': self.voxels[idx].shape,
            'num_occupied': np.sum(self.voxels[idx] > 0),
        }
    
    @property
    def classes(self) -> List[str]:
        """Get list of class names."""
        return self.CLASSES


def create_dataloaders(
    config: Dict[str, Any],
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        batch_size: Override batch size from config
        num_workers: Override number of workers
        
    Returns:
        train_loader, val_loader
    """
    data_config = config['data']
    training_config = config['training']
    
    # Parameters
    batch_size = batch_size or training_config['batch_size']
    num_workers = num_workers or data_config['num_workers']
    
    # Create datasets
    train_dataset = ModelNet10Voxels(
        root_dir=data_config['root_dir'],
        split='train',
        voxel_size=data_config['voxel_size'],
        config=config
    )
    
    test_dataset = ModelNet10Voxels(
        root_dir=data_config['root_dir'],
        split='test',
        voxel_size=data_config['voxel_size'],
        config=config
    )
    
    # Split train into train/val
    val_split = training_config['validation_split']
    n_val = int(len(train_dataset) * val_split)
    n_train = len(train_dataset) - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(training_config['seed'])
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=data_config.get('prefetch_factor', 2),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader 