"""Evaluation metrics for 3D shape classification"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path

# Lazy imports for matplotlib/seaborn to avoid font issues
plt = None
sns = None


def calculate_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    average: str = 'macro',
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Model predictions (class indices or logits)
        targets: Ground truth labels
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        if predictions.dim() > 1:  # Logits
            predictions = predictions.argmax(dim=-1)
        predictions = predictions.detach().cpu().numpy()
    
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average=average, zero_division=0
    )
    
    # Per-class metrics
    per_class_accuracy = []
    unique_classes = np.unique(targets)
    for cls in unique_classes:
        mask = targets == cls
        if mask.sum() > 0:
            cls_acc = (predictions[mask] == targets[mask]).mean()
            per_class_accuracy.append(cls_acc)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_accuracy': np.mean(per_class_accuracy),
    }
    
    return metrics


def plot_confusion_matrix(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
) -> Tuple[Any, np.ndarray]:
    """
    Plot confusion matrix.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        save_path: Path to save figure
        show: Whether to display
        figsize: Figure size
        cmap: Colormap
        
    Returns:
        Figure and confusion matrix array
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)
        predictions = predictions.detach().cpu().numpy()
    
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Lazy import matplotlib/seaborn
    global plt, sns
    if plt is None:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
    if sns is None:
        import seaborn as sns
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        cbar_kws={"shrink": 0.8},
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    
    # Labels
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig, cm


def get_classification_report(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    class_names: List[str],
    save_path: Optional[str] = None,
) -> str:
    """
    Generate detailed classification report.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        class_names: Names of classes
        save_path: Path to save report
        
    Returns:
        Classification report string
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)
        predictions = predictions.detach().cpu().numpy()
    
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Generate report
    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        digits=3,
    )
    
    # Save if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report


def calculate_per_class_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_classes: int,
) -> Dict[str, np.ndarray]:
    """
    Calculate per-class metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with per-class metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)
        predictions = predictions.detach().cpu().numpy()
    
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Initialize arrays
    class_accuracy = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_f1 = np.zeros(num_classes)
    class_support = np.zeros(num_classes)
    
    # Calculate per-class metrics
    for cls in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((predictions == cls) & (targets == cls)).sum()
        fp = ((predictions == cls) & (targets != cls)).sum()
        fn = ((predictions != cls) & (targets == cls)).sum()
        tn = ((predictions != cls) & (targets != cls)).sum()
        
        # Support (number of true instances)
        class_support[cls] = (targets == cls).sum()
        
        # Accuracy
        if class_support[cls] > 0:
            class_accuracy[cls] = (predictions[targets == cls] == cls).mean()
        
        # Precision
        if tp + fp > 0:
            class_precision[cls] = tp / (tp + fp)
        
        # Recall
        if tp + fn > 0:
            class_recall[cls] = tp / (tp + fn)
        
        # F1 score
        if class_precision[cls] + class_recall[cls] > 0:
            class_f1[cls] = 2 * (class_precision[cls] * class_recall[cls]) / \
                           (class_precision[cls] + class_recall[cls])
    
    return {
        'accuracy': class_accuracy,
        'precision': class_precision,
        'recall': class_recall,
        'f1_score': class_f1,
        'support': class_support,
    }


def plot_per_class_metrics(
    per_class_metrics: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
) -> Any:
    """
    Plot per-class metrics as grouped bar chart.
    
    Args:
        per_class_metrics: Dictionary of per-class metrics
        class_names: Names of classes
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    # Lazy import matplotlib
    global plt
    if plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(class_names))
    width = 0.2
    
    # Plot bars
    for i, metric in enumerate(metrics):
        values = per_class_metrics[metric]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    # Customize plot
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    # Add value labels on bars
    for i, metric in enumerate(metrics):
        values = per_class_metrics[metric]
        for j, v in enumerate(values):
            if v > 0:
                ax.text(j + i * width, v + 0.01, f'{v:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def calculate_model_complexity(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Calculate model complexity metrics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with complexity metrics
    """
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    # Calculate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'parameter_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024,
    } 