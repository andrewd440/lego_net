#!/bin/bash
set -e

echo "🚀 Setting up 3D Shape Classification Environment for macOS (Fixed)"
echo "============================================================"
echo "This script resolves OpenMP conflicts by using Apple's Accelerate framework"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: Conda is not installed. Please install Anaconda or Miniconda first.${NC}"
    exit 1
fi

# Remove existing environment if it exists
echo -e "${YELLOW}Checking for existing torch3d environment...${NC}"
if conda env list | grep -q "torch3d"; then
    echo -e "${YELLOW}Found existing torch3d environment. Removing...${NC}"
    conda env remove -n torch3d -y
fi

# Create environment from yml file
echo -e "${GREEN}Creating conda environment from environment.yml...${NC}"
conda env create -f environment.yml

# Initialize conda for current shell
eval "$(conda shell.bash hook)"

# Activate environment
conda activate torch3d

# Verify installation
echo -e "${GREEN}Verifying installation...${NC}"
python -c "
import sys
import platform
print('='*60)
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.machine()}')
print('='*60)
"

# Test PyTorch installation
echo -e "${GREEN}Testing PyTorch installation...${NC}"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS Available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print(f'MPS Built: {torch.backends.mps.is_built()}')
    # Test basic operation
    x = torch.randn(3, 3).to('mps')
    print(f'Successfully created tensor on MPS: {x.device}')
"

# Test critical imports
echo -e "${GREEN}Testing critical imports...${NC}"
python -c "
print('Testing imports...')
try:
    import numpy as np
    print('✓ NumPy')
    import scipy
    print('✓ SciPy')
    import sklearn
    print('✓ Scikit-learn')
    import open3d
    print('✓ Open3D')
    import torch_geometric
    print('✓ PyTorch Geometric')
    import matplotlib
    matplotlib.use('Agg')  # Set backend
    import matplotlib.pyplot as plt
    print('✓ Matplotlib')
    import plotly
    print('✓ Plotly')
    print('All imports successful!')
except ImportError as e:
    print(f'Import error: {e}')
    exit(1)
"

# Setup Jupyter kernel
echo -e "${GREEN}Setting up Jupyter kernel...${NC}"
python -m ipykernel install --user --name torch3d --display-name "PyTorch 3D (Metal)"

# Create project directories
echo -e "${GREEN}Creating project directories...${NC}"
mkdir -p data/ModelNet10 data/cache
mkdir -p checkpoints logs plots outputs
mkdir -p notebooks tests

# Test the evaluation script
echo -e "${GREEN}Testing evaluation script compatibility...${NC}"
python -c "
# Test that we can import everything needed for evaluation
try:
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    print('✓ All evaluation dependencies working correctly')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

echo -e "${GREEN}✅ Environment setup complete!${NC}"
echo ""
echo "Key improvements in this setup:"
echo "- Uses Apple's Accelerate framework instead of OpenBLAS"
echo "- Avoids OpenMP conflicts entirely"
echo "- Ensures all PyTorch dependencies are compatible"
echo ""
echo "To activate the environment, run:"
echo -e "${GREEN}conda activate torch3d${NC}"
echo ""
echo "To run the evaluation script:"
echo -e "${GREEN}python evaluate_model.py --model_path checkpoints/best_model.pt --num_vis 5${NC}"
echo ""
echo "If you still encounter issues, you can add this to your .zshrc:"
echo "export KMP_DUPLICATE_LIB_OK=TRUE" 