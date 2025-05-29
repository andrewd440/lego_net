import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

from src.models.voxel_cnn import VoxelCNN
from src.data.dataset import ModelNet10Voxels, create_dataloaders
from src.utils.config import load_config


def evaluate_validation(model_path, config_path='configs/default.yaml'):
    """Evaluate model on validation split (from training data)"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders (same as training)
    train_loader, val_loader = create_dataloaders(config)
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Load model
    model = VoxelCNN(config).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with saved val_acc: {checkpoint['val_acc']:.4f}")
    
    # Evaluation
    correct = 0
    total = 0
    
    with torch.no_grad():
        for voxels, labels in tqdm(val_loader, desc="Evaluating validation set"):
            voxels = voxels.to(device)
            labels = labels.to(device)
            
            # Model expects channel dimension
            if voxels.dim() == 4:
                voxels = voxels.unsqueeze(1)
            
            outputs = model(voxels)
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100. * correct / total
    print(f"\nValidation Accuracy (recalculated): {accuracy:.2f}% ({correct}/{total})")
    print(f"Matches saved accuracy: {'YES' if abs(accuracy - checkpoint['val_acc']) < 0.1 else 'NO'}")
    
    # Now evaluate on test set for comparison
    print("\n" + "="*50)
    print("Now evaluating on TEST set for comparison:")
    
    test_dataset = ModelNet10Voxels(
        root_dir=config['data']['root_dir'],
        split='test',
        voxel_size=config['data']['voxel_size']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['testing']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for voxels, labels in tqdm(test_loader, desc="Evaluating test set"):
            voxels = voxels.to(device)
            labels = labels.to(device)
            
            if voxels.dim() == 4:
                voxels = voxels.unsqueeze(1)
            
            outputs = model(voxels)
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    test_accuracy = 100. * correct / total
    print(f"\nTest Accuracy: {test_accuracy:.2f}% ({correct}/{total})")
    print(f"Difference (Val - Test): {accuracy - test_accuracy:.2f}%")
    
    return accuracy, test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                       default='configs/default.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    val_acc, test_acc = evaluate_validation(args.model_path, args.config) 