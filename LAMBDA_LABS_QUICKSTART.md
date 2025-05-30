# Lambda Labs Deployment Guide

## Quick Start

1. **SSH into your Lambda Labs instance**
   ```bash
   ssh ubuntu@<your-instance-ip>
   ```

2. **Clone and setup the repository**
   ```bash
   git clone <your-repo-url>
   cd lego_net
   chmod +x setup_lambda_labs.sh
   ./setup_lambda_labs.sh
   ```

3. **Activate the environment**
   ```bash
   conda activate torch3d
   ```

4. **Start training in a persistent session**
   ```bash
   # Start tmux session
   tmux new -s training
   
   # Run training with Lambda Labs config
   python src/train.py --config configs/lambda_labs.yaml
   
   # Detach from tmux: Ctrl+B, then D
   ```

5. **Monitor training**
   ```bash
   # GPU usage
   watch -n 1 nvidia-smi
   
   # Training logs (in another terminal)
   tensorboard --logdir logs --host 0.0.0.0 --port 6006
   ```

## Key Differences from Local Setup

### 1. Device Configuration
- Uses `cuda` instead of `mps` (Apple Silicon)
- Supports mixed precision training for faster performance
- Larger batch sizes (64 vs 32) due to more GPU memory

### 2. Performance Optimizations
- Increased number of data workers (8 vs 4)
- Higher prefetch factor for better GPU utilization
- Mixed precision training enabled

### 3. Remote Monitoring
- Weights & Biases (wandb) enabled by default
- TensorBoard accessible via `http://<instance-ip>:6006`

## GPU Memory Guidelines

Adjust batch size based on your GPU:

| GPU Type | Recommended Batch Size | Max Batch Size |
|----------|----------------------|----------------|
| A10 (24GB) | 64 | 128 |
| A100 (40GB) | 128 | 256 |
| A100 (80GB) | 256 | 512 |
| H100 (80GB) | 256 | 512 |

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in configs/lambda_labs.yaml
# or pass it as argument:
python src/train.py --config configs/lambda_labs.yaml --batch_size 32
```

### Dataset Download Issues
```bash
# Manually download if automatic download fails
cd data
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip ModelNet10.zip
cd ..
```

### Permission Denied Errors
```bash
# If you get permission errors
sudo chown -R $USER:$USER .
```

## Cost-Saving Tips

1. **Use spot instances** for experimentation
2. **Save checkpoints frequently** (already configured)
3. **Use early stopping** to avoid overtraining
4. **Monitor via W&B** instead of keeping SSH open
5. **Terminate instance** when training completes

## Advanced Usage

### Resume Training
```bash
python src/train.py --config configs/lambda_labs.yaml --resume checkpoints/best_model.pt
```

### Multi-GPU Training (if available)
```bash
python -m torch.distributed.launch --nproc_per_node=<num_gpus> src/train.py --config configs/lambda_labs.yaml
```

### Custom Configurations
Create your own config by copying and modifying:
```bash
cp configs/lambda_labs.yaml configs/my_config.yaml
# Edit my_config.yaml
python src/train.py --config configs/my_config.yaml
``` 