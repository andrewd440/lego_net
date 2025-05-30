#!/bin/bash
set -e

echo "🚀 Setting up 3D Shape Classification Environment for macOS M2 (Conda-focused)"
echo "============================================================"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: Conda is not installed. Please install Anaconda or Miniconda first.${NC}"
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if we're on macOS with Apple Silicon
if [[ $(uname) == "Darwin" ]] && [[ $(uname -m) == "arm64" ]]; then
    echo -e "${GREEN}✓ Detected macOS with Apple Silicon (M1/M2/M3)${NC}"
else
    echo -e "${YELLOW}Warning: This script is optimized for macOS M2. Proceeding anyway...${NC}"
fi

# Remove existing environment if it exists
if conda env list | grep -q "^torch3d "; then
    echo -e "${YELLOW}Found existing 'torch3d' environment. Removing...${NC}"
    conda deactivate 2>/dev/null || true
    conda env remove -n torch3d -y
fi

# Create conda environment from YAML file
echo -e "${GREEN}Creating conda environment from environment.yml...${NC}"
conda env create -f environment.yml

# Initialize conda for the current shell
eval "$(conda shell.bash hook)"

# Activate environment
echo -e "${GREEN}Activating environment...${NC}"
conda activate torch3d

# Verify Metal Performance Shaders availability
echo -e "${GREEN}Verifying Metal Performance Shaders (MPS) support...${NC}"
python -c "
import torch
import platform
import sys

print('='*60)
print(f'Python: {sys.version.split()[0]}')
print(f'System: {platform.system()} {platform.machine()}')
print(f'PyTorch version: {torch.__version__}')
print(f'MPS Available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print(f'MPS Built: {torch.backends.mps.is_built()}')
    # Test MPS functionality
    try:
        device = torch.device('mps')
        x = torch.randn(10, 10, device=device)
        y = x @ x.T
        print(f'MPS tensor operations: ✓ Working')
    except Exception as e:
        print(f'MPS tensor operations: ✗ Error - {e}')
print('='*60)
"

# Verify key packages
echo -e "${GREEN}Verifying key package installations...${NC}"
python -c "
import importlib
import sys

packages = {
    'Core': ['numpy', 'scipy', 'sklearn', 'pandas'],
    'PyTorch': ['torch', 'torchvision', 'torchaudio'],
    '3D Processing': ['open3d', 'trimesh', 'pyvoxel'],
    'Visualization': ['matplotlib', 'seaborn', 'plotly'],
    'ML Tools': ['tensorboard', 'wandb', 'tqdm'],
    'Development': ['pytest', 'black', 'isort']
}

print('\\nPackage Installation Status:')
print('-' * 40)

all_good = True
for category, pkgs in packages.items():
    print(f'\\n{category}:')
    for pkg in pkgs:
        try:
            if pkg == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(pkg)
            print(f'  ✓ {pkg}')
        except ImportError as e:
            print(f'  ✗ {pkg} - {e}')
            all_good = False

print('\\n' + '='*40)
if all_good:
    print('✅ All packages installed successfully!')
else:
    print('⚠️  Some packages failed to install')
    sys.exit(1)
"

# Setup Jupyter kernel
echo -e "${GREEN}Setting up Jupyter kernel...${NC}"
python -m ipykernel install --user --name torch3d --display-name "PyTorch 3D (M2 Metal)"

# Configure nbdime for better notebook diffs (only if in git repo)
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${GREEN}Configuring nbdime for notebook version control...${NC}"
    nbdime config-git --enable --global
    echo -e "${GREEN}nbdime configured for git${NC}"
fi

# Create a simple test script
echo -e "${GREEN}Creating test script...${NC}"
cat > test_mps_3d.py << 'EOF'
#!/usr/bin/env python
"""Quick test of 3D operations on MPS"""
import torch
import torch.nn as nn

# Test 3D convolution on MPS
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a simple 3D conv layer
conv3d = nn.Conv3d(1, 16, kernel_size=3, padding=1).to(device)
x = torch.randn(1, 1, 32, 32, 32).to(device)

try:
    output = conv3d(x)
    print(f"✓ 3D Convolution works! Output shape: {output.shape}")
except Exception as e:
    print(f"✗ 3D Convolution failed: {e}")
    print("Note: Some 3D operations may not be supported on MPS yet")
EOF

echo -e "${GREEN}✅ Environment setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Activate the environment: ${GREEN}conda activate torch3d${NC}"
echo "2. Test MPS support: ${GREEN}python test_mps_3d.py${NC}"
echo "3. Run the full test suite: ${GREEN}python test_setup.py${NC}"
echo "4. Start training: ${GREEN}python src/train.py --config configs/default.yaml${NC}"
echo ""
echo "To update packages later:"
echo "  - Update all conda packages: ${GREEN}conda update --all${NC}"
echo "  - Update pip packages: ${GREEN}pip list --outdated${NC}" 