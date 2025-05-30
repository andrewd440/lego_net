# Environment Setup Guide for macOS M2

This guide provides detailed instructions for setting up the development environment for the 3D Shape Classification project on macOS M2 (Apple Silicon).

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Conda (Miniconda or Anaconda) installed
- At least 8GB of free disk space
- Git (for version control features)

## Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd lego_net

# Run the setup script
chmod +x setup_env_conda.sh
./setup_env_conda.sh
```

## Manual Setup

If you prefer to set up the environment manually:

```bash
# Create environment from YAML
conda env create -f environment.yml

# Activate environment
conda activate torch3d

# Verify installation
python test_setup.py
```

## Dependencies Overview

### Core Scientific Computing
- **numpy>=2.2,<3.0**: Array operations (explicit version for PyTorch 2.7 compatibility)
- **scipy**: Scientific computing utilities
- **scikit-learn**: Machine learning metrics and utilities
- **pandas**: Data manipulation

### Deep Learning Framework
- **pytorch=2.7.0**: Core deep learning framework (pinned version)
- **torchvision**: Computer vision utilities
- **torchaudio**: Audio processing (included for completeness)

### 3D Processing
- **open3d**: 3D data processing, mesh loading, and voxelization
- **torch-geometric**: Graph neural networks (for future extensions)

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive 3D visualization

### Development Tools
- **jupyter/jupyterlab**: Interactive development
- **tensorboard**: Training visualization
- **wandb**: Experiment tracking
- **pytest**: Testing framework
- **black**: Code formatting

## Version Compatibility Notes

### NumPy 2.0+ Compatibility
PyTorch 2.7.0 was compiled against NumPy 2.2+ API. Using older NumPy versions (1.x) will cause segmentation faults due to ABI incompatibility. Our environment.yml ensures NumPy >=2.2,<3.0 is installed.

### PyTorch Version Pinning
We pin PyTorch to version 2.7.0 to ensure:
- Consistent behavior across installations
- NumPy 2.0+ compatibility
- Python 3.13 support
- Latest performance optimizations

## Troubleshooting

### NumPy Compatibility Issues
If you encounter segmentation faults or NumPy-related errors:

```bash
# Check NumPy version
python -c "import numpy; print(numpy.__version__)"

# Should be 2.2.0 or higher
# If not, update NumPy:
conda install "numpy>=2.2,<3.0" -y
```

### Metal Performance Shaders Not Available
- Ensure you're running on Apple Silicon (M1/M2/M3)
- Update macOS to latest version
- Verify PyTorch installation: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Out of Memory Errors
- Reduce batch size in configuration
- Enable gradient checkpointing
- Use mixed precision training

### Slow Training
- Verify MPS is being used: check device in logs
- Increase number of data loader workers
- Disable data augmentation for testing

## Environment Recreation

If you need to recreate the environment (e.g., after dependency conflicts):

```bash
# Remove existing environment
conda env remove -n torch3d

# Recreate from updated YAML
conda env create -f environment.yml

# Activate and test
conda activate torch3d
python test_setup.py
```

## Performance Optimization

### Apple Silicon Optimizations
- Uses Apple's Accelerate framework for BLAS operations
- Metal Performance Shaders for GPU acceleration
- Optimized memory allocation for M-series chips

### Conda vs Pip
This setup prioritizes conda packages for better dependency resolution and Apple Silicon optimization. Only packages unavailable in conda (like Open3D) are installed via pip.

## Security Notes

Starting with PyTorch 2.6, the default value for `weights_only` parameter in `torch.load` has changed for security reasons. This may affect model loading in older code.

## Support

For issues specific to this environment setup:
1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Try recreating the environment from scratch
4. Report persistent issues with full error logs 