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
    cd ~
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    
    # Add conda to PATH
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    source ~/.bashrc
    
    # Clean up
    rm miniconda.sh
    cd -
fi

# Ensure conda is available
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh

# Create environment from yml file
echo "Creating conda environment..."
conda env create -f environment.yml --force

# Activate environment
echo "Activating environment..."
conda activate torch3d

# Install CUDA-specific PyTorch (override the conda version for better CUDA support)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install missing packages that might not be in conda
echo "Installing additional packages..."
pip install open3d==0.19.0 plotly tqdm h5py wandb tensorboard

# Test CUDA availability
echo "Testing CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Test imports
echo "Testing package imports..."
python -c "
try:
    import open3d; print('✓ Open3D imported successfully')
except ImportError as e: print(f'✗ Open3D import failed: {e}')

try:
    import plotly; print('✓ Plotly imported successfully') 
except ImportError as e: print(f'✗ Plotly import failed: {e}')

try:
    import tqdm; print('✓ TQDM imported successfully')
except ImportError as e: print(f'✗ TQDM import failed: {e}')

try:
    import h5py; print('✓ H5Py imported successfully')
except ImportError as e: print(f'✗ H5Py import failed: {e}')
"

# Download dataset if not present
if [ ! -d "data/ModelNet10" ]; then
    echo "Downloading ModelNet10 dataset..."
    python -c "from src.data.dataset import ModelNet10Voxels; ModelNet10Voxels(root_dir='./data/ModelNet10', split='train', download=True)"
fi

# Create necessary directories
mkdir -p checkpoints logs plots outputs

# Install tmux for persistent sessions
sudo apt-get update && sudo apt-get install -y tmux htop

# Create activation script for easy use
echo "Creating activation script..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate torch3d
echo "Environment activated! Current Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
EOF
chmod +x activate_env.sh

echo "Setup complete! To start training:"
echo "  1. Activate environment: source activate_env.sh"
echo "  2. Test setup: python test_setup.py"
echo "  3. Start tmux: tmux new -s training"
echo "  4. Run training: python src/train.py --config configs/lambda_labs.yaml"
echo "  5. Detach from tmux: Ctrl+B, then D"
echo "  6. Monitor GPU: nvidia-smi -l 1"
echo "  7. Monitor training: tensorboard --logdir logs --host 0.0.0.0" 