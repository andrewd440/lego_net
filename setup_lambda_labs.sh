#!/bin/bash
# Setup script for Lambda Labs GPU instances

echo "Setting up LEGO Net on Lambda Labs..."

# Check if running on Lambda Labs (they typically have nvidia-smi)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Are you sure this is a GPU instance?"
fi

# Display GPU info
echo "GPU Information:"
nvidia-smi

# Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init
    source ~/.bashrc
fi

# Create environment from yml file
echo "Creating conda environment..."
conda env create -f environment.yml

# Activate environment
echo "Activating environment..."
conda activate torch3d

# Install CUDA-specific PyTorch (override the conda version for better CUDA support)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Download dataset if not present
if [ ! -d "data/ModelNet10" ]; then
    echo "Downloading ModelNet10 dataset..."
    python -c "from src.data.dataset import ModelNet10Voxels; ModelNet10Voxels(root_dir='./data/ModelNet10', split='train', download=True)"
fi

# Create necessary directories
mkdir -p checkpoints logs plots outputs

# Install tmux for persistent sessions
sudo apt-get update && sudo apt-get install -y tmux htop

echo "Setup complete! To start training:"
echo "  1. Start a tmux session: tmux new -s training"
echo "  2. Run training: python src/train.py --config configs/lambda_labs.yaml"
echo "  3. Detach from tmux: Ctrl+B, then D"
echo "  4. Monitor GPU usage: nvidia-smi -l 1"
echo "  5. Monitor training: tensorboard --logdir logs --host 0.0.0.0" 