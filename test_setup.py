#!/usr/bin/env python3
"""Test script to verify environment setup and basic functionality"""

import sys
import os

# Set matplotlib backend before importing to avoid font issues
os.environ['MPLBACKEND'] = 'Agg'

# Prevent OpenBLAS threading conflicts that cause Open3D segfaults on macOS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from pathlib import Path
import time

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
ENDC = '\033[0m'


def print_status(message, status='info'):
    """Print colored status message"""
    if status == 'success':
        print(f"{GREEN}✓ {message}{ENDC}")
    elif status == 'error':
        print(f"{RED}✗ {message}{ENDC}")
    elif status == 'warning':
        print(f"{YELLOW}⚠ {message}{ENDC}")
    else:
        print(f"{BLUE}→ {message}{ENDC}")


def test_imports():
    """Test all required imports"""
    print("\n1. Testing imports...")    
    
    modules = [
        ('open3d', 'Open3D'),  # Import Open3D FIRST before NumPy
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),  # NumPy after Open3D
        ('plotly', 'Plotly'),
        ('h5py', 'H5Py'),
        ('tqdm', 'TQDM'),
        ('sklearn', 'Scikit-learn'),
    ]
    
    all_good = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print_status(f"{display_name} imported successfully", 'success')
        except ImportError as e:
            print_status(f"Failed to import {display_name}: {e}", 'error')
            all_good = False
    
    return all_good


def test_metal_support():
    """Test Metal Performance Shaders support"""
    print("\n2. Testing Metal support...")
    
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print_status("Metal Performance Shaders available", 'success')
        
        if torch.backends.mps.is_built():
            print_status("MPS backend is built", 'success')
        else:
            print_status("MPS backend not built", 'warning')
        
        # Test MPS tensor creation
        try:
            device = torch.device('mps')
            x = torch.randn(10, 10, device=device)
            y = torch.randn(10, 10, device=device)
            z = torch.matmul(x, y)
            print_status(f"MPS tensor operations working (result shape: {z.shape})", 'success')
            return True
        except Exception as e:
            print_status(f"MPS tensor operations failed: {e}", 'error')
            return False
    else:
        print_status("Metal Performance Shaders not available", 'warning')
        print_status("Will use CPU for computation", 'info')
        return False


def test_project_structure():
    """Test project directory structure"""
    print("\n3. Testing project structure...")
    
    required_dirs = [
        'configs',
        'src/models',
        'src/data',
        'src/utils',
        'notebooks',
        'tests',
        'data/ModelNet10',
        'checkpoints',
        'logs',
    ]
    
    all_good = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print_status(f"Directory '{dir_path}' exists", 'success')
        else:
            print_status(f"Directory '{dir_path}' missing", 'error')
            all_good = False
    
    return all_good


def test_config_loading():
    """Test configuration loading"""
    print("\n4. Testing configuration...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from src.utils.config import load_config
        
        config_path = Path('configs/default.yaml')
        if not config_path.exists():
            print_status("Config file not found", 'error')
            return False
        
        config = load_config('configs/default.yaml')
        print_status("Configuration loaded successfully", 'success')
        
        # Check key sections
        required_sections = ['model', 'data', 'training', 'optimizer']
        for section in required_sections:
            if section in config:
                print_status(f"  - {section} section present", 'success')
            else:
                print_status(f"  - {section} section missing", 'error')
                return False
        
        return True
    except Exception as e:
        print_status(f"Configuration test failed: {e}", 'error')
        return False


def test_voxelization():
    """Test basic voxelization functionality"""
    print("\n5. Testing voxelization...")
    
    try:
        from src.data.voxelization import voxelize_point_cloud
        
        # Create random point cloud
        points = np.random.randn(1000, 3)
        
        # Test voxelization
        start_time = time.time()
        voxels = voxelize_point_cloud(points, voxel_size=32)
        elapsed = time.time() - start_time
        
        print_status(f"Voxelization successful (shape: {voxels.shape}, time: {elapsed:.3f}s)", 'success')
        
        # Check output
        if voxels.shape == (32, 32, 32):
            print_status("Output shape correct", 'success')
        else:
            print_status(f"Unexpected output shape: {voxels.shape}", 'error')
            return False
        
        occupied = (voxels > 0).sum().item()
        print_status(f"Occupied voxels: {occupied}/{32**3}", 'info')
        
        return True
    except Exception as e:
        print_status(f"Voxelization test failed: {e}", 'error')
        return False


def test_model_creation():
    """Test model creation"""
    print("\n6. Testing model creation...")
    
    try:
        from src.models import create_model
        from src.utils.config import load_config
        
        # Load config
        config = load_config('configs/default.yaml')
        
        # Create model
        model = create_model(config)
        print_status("Model created successfully", 'success')
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print_status(f"Total parameters: {total_params:,}", 'info')
        print_status(f"Trainable parameters: {trainable_params:,}", 'info')
        
        # Test forward pass
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        dummy_input = torch.randn(2, 1, 32, 32, 32)
        
        # Move model to device
        model = model.to(device)
        dummy_input = dummy_input.to(device)
        
        start_time = time.time()
        output = model(dummy_input)
        elapsed = time.time() - start_time
        
        print_status(f"Forward pass successful (output shape: {output.shape}, time: {elapsed:.3f}s)", 'success')
        
        if output.shape == (2, 10):  # batch_size=2, num_classes=10
            print_status("Output shape correct", 'success')
            return True
        else:
            print_status(f"Unexpected output shape: {output.shape}", 'error')
            return False
        
    except Exception as e:
        print_status(f"Model test failed: {e}", 'error')
        return False


def test_data_loading():
    """Test data loading (without downloading)"""
    print("\n7. Testing data loading setup...")
    
    try:
        from src.data.dataset import ModelNet10Voxels
        
        # Test dataset initialization (without download)
        print_status("Checking ModelNet10 dataset class...", 'info')
        
        # Just verify the class can be imported
        print_status("Dataset class imported successfully", 'success')
        print_status("Note: Actual dataset will be downloaded on first run", 'warning')
        
        return True
    except Exception as e:
        print_status(f"Data loading test failed: {e}", 'error')
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("3D Shape Classification Environment Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Metal Support", test_metal_support),
        ("Project Structure", test_project_structure),
        ("Configuration", test_config_loading),
        ("Voxelization", test_voxelization),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_status(f"Test '{test_name}' crashed: {e}", 'error')
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = 'success' if result else 'error'
        symbol = '✓' if result else '✗'
        print_status(f"{test_name}: {symbol}", status)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print_status("\nAll tests passed! Environment is ready.", 'success')
        print("\nNext steps:")
        print("1. Run training: python src/train.py --config configs/default.yaml")
        print("2. Open notebooks: jupyter lab")
        return 0
    else:
        print_status(f"\n{total-passed} tests failed. Please check the errors above.", 'error')
        return 1


if __name__ == "__main__":
    sys.exit(main()) 