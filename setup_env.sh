#!/bin/bash
set -e

echo "🚀 Setting up 3D Shape Classification Environment for macOS with Metal Support"
echo "============================================================"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: Conda is not installed. Please install Anaconda or Miniconda first.${NC}"
    exit 1
fi

# Create conda environment
echo -e "${GREEN}Creating conda environment 'torch3d'...${NC}"
conda create -n torch3d python=3.10 -y

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate environment
conda activate torch3d

# Install PyTorch with Metal support
echo -e "${GREEN}Installing PyTorch with Metal Performance Shaders support...${NC}"
pip install torch torchvision torchaudio

# Install PyTorch Geometric and dependencies
echo -e "${GREEN}Installing PyTorch Geometric...${NC}"
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric==2.5.0

# Install 3D processing and visualization libraries
echo -e "${GREEN}Installing 3D processing libraries...${NC}"
pip install open3d==0.17.0
pip install plotly==5.17.0
pip install trimesh==4.0.4
pip install pyvoxel

# Install ML and utility packages
echo -e "${GREEN}Installing ML utilities...${NC}"
pip install scikit-learn==1.3.0
pip install pyyaml==6.0
pip install tqdm==4.66.0
pip install h5py==3.10.0
pip install tensorboard==2.15.0

# Install development tools
echo -e "${GREEN}Installing development tools...${NC}"
pip install jupyter==1.0.0
pip install ipywidgets==8.1.1
pip install nbdime==3.2.1
pip install wandb==0.15.12
pip install pytest==7.4.3
pip install pytest-cov==4.1.0
pip install black==23.11.0
pip install isort==5.12.0

# Verify Metal availability
echo -e "${GREEN}Verifying Metal Performance Shaders availability...${NC}"
python -c "
import torch
import platform
print('='*60)
print(f'System: {platform.system()} {platform.machine()}')
print(f'PyTorch version: {torch.__version__}')
print(f'MPS Available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print(f'MPS Built: {torch.backends.mps.is_built()}')
print('='*60)
"

# Create requirements.txt
echo -e "${GREEN}Creating requirements.txt...${NC}"
pip freeze > requirements.txt

# Setup Jupyter kernel
echo -e "${GREEN}Setting up Jupyter kernel...${NC}"
python -m ipykernel install --user --name torch3d --display-name "PyTorch 3D (Metal)"

# Configure nbdime for better notebook diffs (only if in git repo)
echo -e "${GREEN}Configuring nbdime for notebook version control...${NC}"
if git rev-parse --git-dir > /dev/null 2>&1; then
    nbdime config-git --enable
    echo -e "${GREEN}nbdime configured for git${NC}"
else
    echo -e "${YELLOW}Warning: Not in a git repository. Skipping nbdime git configuration.${NC}"
    echo -e "${YELLOW}To enable notebook diffs later, run: nbdime config-git --enable${NC}"
fi

echo -e "${GREEN}✅ Environment setup complete!${NC}"
echo -e "${GREEN}To activate the environment, run: conda activate torch3d${NC}"
echo ""
echo "Next steps:"
echo "1. Run the test script: python test_setup.py"
echo "2. Start training: python src/train.py --config configs/default.yaml"
echo "3. Open notebooks: jupyter lab" 