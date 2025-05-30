import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
from tqdm import tqdm
import argparse
import math
from pathlib import Path

from src.models.voxel_cnn import VoxelCNN
from src.data.dataset import ModelNet10Voxels
from src.utils.config import load_config


def evaluate_model(model_path, config_path='configs/default.yaml', num_visualizations=10):
    """Evaluate trained model and generate detailed analysis"""
    
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
    all_probs = []
    
    # Debug first batch
    first_batch = True
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            voxels, labels = batch
            voxels = voxels.to(device)
            labels = labels.to(device)
            
            # Debug info for first batch
            if first_batch:
                print(f"\nDebug - Voxel shape: {voxels.shape}")
                print(f"Debug - Voxel range: [{voxels.min():.3f}, {voxels.max():.3f}]")
                print(f"Debug - Labels: {labels[:5]}")
                first_batch = False
            
            outputs = model(voxels)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")
    
    # Class names
    class_names = test_dataset.classes
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300)
    plt.close()
    
    # Classification Report
    report = classification_report(all_labels, all_preds, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Save detailed report
    with open('outputs/classification_report.txt', 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, 
                                    target_names=class_names))
    
    # Per-class accuracy visualization
    per_class_acc = [report[cls]['recall'] for cls in class_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, per_class_acc)
    plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall: {accuracy:.3f}')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('outputs/per_class_accuracy.png', dpi=300)
    plt.close()
    
    # Visualize predictions
    visualize_predictions(model, test_dataset, device, 
                         num_samples=num_visualizations)
    
    # Find most confused pairs
    analyze_confusion_pairs(cm, class_names)
    
    return accuracy, report


def visualize_predictions(model, dataset, device, num_samples=10):
    """Visualize model predictions on random samples"""
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Calculate grid dimensions (make sure we have enough subplots)
    cols = 5  # Fixed number of columns
    rows = math.ceil(num_samples / cols)  # Calculate required rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.ravel()  # Flatten the array for easier indexing
    
    for idx, i in enumerate(indices):
        voxel, label = dataset[i]
        voxel = voxel.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(voxel)
            probs = torch.softmax(output, dim=1).squeeze()
            pred = output.argmax(dim=1).item()
        
        # Create 2D projection of voxel
        voxel_np = voxel.squeeze().cpu().numpy()
        projection = voxel_np.max(axis=0)  # Max projection along z-axis
        
        axes[idx].imshow(projection, cmap='gray')
        axes[idx].set_title(f'True: {dataset.classes[label]}\n'
                           f'Pred: {dataset.classes[pred]} '
                           f'({probs[pred]:.2f})',
                           color='green' if pred == label else 'red')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
        axes[idx].set_visible(False)
    
    plt.suptitle('Model Predictions (Green=Correct, Red=Wrong)')
    plt.tight_layout()
    plt.savefig('outputs/sample_predictions.png', dpi=300)
    plt.close()


def analyze_confusion_pairs(cm, class_names):
    """Analyze most confused class pairs"""
    
    # Set diagonal to 0 to find off-diagonal maxima
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    
    # Find top confused pairs
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_copy[i, j] > 0:
                confused_pairs.append((cm_copy[i, j], i, j))
    
    confused_pairs.sort(reverse=True)
    
    print("\nMost Confused Class Pairs:")
    for count, i, j in confused_pairs[:5]:
        print(f"  {class_names[i]} -> {class_names[j]}: {count} misclassifications")


def create_3d_visualization(model, dataset, device, class_idx=0):
    """Create interactive 3D visualization of a sample"""
    
    # Find a sample of the specified class
    for i in range(len(dataset)):
        voxel, label = dataset[i]
        if label == class_idx:
            break
    
    voxel = voxel.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(voxel)
        probs = torch.softmax(output, dim=1).squeeze()
        pred = output.argmax(dim=1).item()
    
    # Create 3D visualization
    voxel_np = voxel.squeeze().cpu().numpy()
    x, y, z = np.where(voxel_np > 0.5)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=z,
            colorscale='Viridis',
        )
    )])
    
    fig.update_layout(
        title=f'{dataset.classes[class_idx]} - Predicted: {dataset.classes[pred]} ({probs[pred]:.2f})',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        )
    )
    
    fig.write_html(f'outputs/3d_visualization_{dataset.classes[class_idx]}.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                       default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--num_vis', type=int, default=10,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    # Run evaluation
    accuracy, report = evaluate_model(
        args.model_path, 
        args.config,
        args.num_vis
    )
    
    # Create 3D visualizations for each class
    config = load_config(args.config)
    dataset = ModelNet10Voxels(
        root_dir=config['data']['root_dir'],
        split='test',
        voxel_size=config['data']['voxel_size']
    )
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    model = VoxelCNN(config).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\nCreating 3D visualizations...")
    for i, class_name in enumerate(dataset.classes):
        create_3d_visualization(model, dataset, device, class_idx=i)
    
    print(f"\nEvaluation complete! Check the 'outputs' directory for results.") 