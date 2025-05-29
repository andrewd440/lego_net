import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import argparse
from pathlib import Path

from src.models.voxel_cnn import VoxelCNN
from src.data.dataset import ModelNet10Voxels
from src.utils.config import load_config


def evaluate_model(model_path, config_path='configs/default.yaml'):
    """Evaluate trained model and print results"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
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
    
    # Load model
    model = VoxelCNN(config).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_acc']:.4f}")
    
    # Evaluation
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (voxels, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            voxels = voxels.to(device)
            labels = labels.to(device)
            
            # Debug first batch
            if batch_idx == 0:
                print(f"\nDebug - Input shape: {voxels.shape}")
                print(f"Debug - Expecting shape: [batch, 32, 32, 32]")
                print(f"Debug - Model expects: [batch, 1, 32, 32, 32]")
            
            # Model expects channel dimension
            if voxels.dim() == 4:
                voxels = voxels.unsqueeze(1)
            
            outputs = model(voxels)
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Classification report
    class_names = test_dataset.classes
    report = classification_report(all_labels, all_preds, 
                                 target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("True\\Pred", end="")
    for i in range(len(class_names)):
        print(f"\t{i}", end="")
    print()
    for i in range(len(class_names)):
        print(f"{i}", end="")
        for j in range(len(class_names)):
            print(f"\t{cm[i,j]}", end="")
        print(f"\t{class_names[i]}")
    
    # Most confused pairs
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm_copy[i, j] > 0:
                confused_pairs.append((cm_copy[i, j], i, j))
    
    confused_pairs.sort(reverse=True)
    print("\nTop 5 Most Confused Pairs:")
    for count, i, j in confused_pairs[:5]:
        print(f"  {class_names[i]} -> {class_names[j]}: {count} misclassifications")
    
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                       default='configs/default.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Run evaluation
    accuracy = evaluate_model(args.model_path, args.config) 