"""3D visualization utilities for voxel grids and point clouds"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Union, List, Tuple, Dict, Any
from pathlib import Path

# Lazy import matplotlib to avoid font issues
plt = None


def visualize_voxels(
    voxels: Union[torch.Tensor, np.ndarray],
    title: str = "3D Voxel Grid",
    threshold: float = 0.5,
    colorscale: str = "Viridis",
    save_path: Optional[str] = None,
    show: bool = True,
    opacity: float = 0.7,
    size: int = 3,
) -> go.Figure:
    """
    Visualize 3D voxel grid using Plotly.
    
    Args:
        voxels: 3D voxel grid tensor/array
        title: Plot title
        threshold: Threshold for binary voxels
        colorscale: Plotly colorscale name
        save_path: Path to save HTML file
        show: Whether to display the plot
        opacity: Opacity of voxels
        size: Size of voxel markers
        
    Returns:
        Plotly figure object
    """
    # Convert to numpy if needed
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.detach().cpu().numpy()
    
    # Get occupied voxel coordinates
    if voxels.ndim == 3:
        occupied = voxels > threshold
    else:
        # Handle multi-channel voxels
        occupied = voxels.any(axis=0) if voxels.shape[0] < voxels.shape[1] else voxels.any(axis=-1)
    
    coords = np.argwhere(occupied)
    
    if len(coords) == 0:
        print("Warning: No occupied voxels found!")
        coords = np.array([[0, 0, 0]])  # Add dummy point
    
    # Create 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=coords[:, 2],  # Color by height
                colorscale=colorscale,
                opacity=opacity,
                line=dict(width=0),
            ),
            text=[f"({x},{y},{z})" for x, y, z in coords],
            hovertemplate="Voxel: %{text}<br>Value: %{marker.color}<extra></extra>",
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
        ),
        showlegend=False,
        width=800,
        height=600,
    )
    
    # Save if requested
    if save_path:
        fig.write_html(save_path)
    
    # Show if requested
    if show:
        fig.show()
    
    return fig


def create_3d_plot(
    points: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "3D Point Cloud",
    save_path: Optional[str] = None,
    show: bool = True,
    size: int = 2,
) -> go.Figure:
    """
    Create 3D scatter plot of points.
    
    Args:
        points: Point coordinates (N, 3)
        colors: Point colors or labels
        title: Plot title
        save_path: Path to save HTML
        show: Whether to display
        size: Point size
        
    Returns:
        Plotly figure
    """
    # Convert to numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    if colors is not None and isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    
    # Create figure
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=colors if colors is not None else points[:, 2],
                colorscale='Viridis',
                showscale=True,
            ),
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data',
        ),
        width=800,
        height=600,
    )
    
    if save_path:
        fig.write_html(save_path)
    
    if show:
        fig.show()
    
    return fig


def plot_voxel_comparison(
    voxels_list: List[Union[torch.Tensor, np.ndarray]],
    titles: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
    threshold: float = 0.5,
) -> go.Figure:
    """
    Plot multiple voxel grids side by side for comparison.
    
    Args:
        voxels_list: List of voxel grids
        titles: List of subplot titles
        save_path: Save path for HTML
        show: Whether to display
        threshold: Voxel threshold
        
    Returns:
        Plotly figure
    """
    n_plots = len(voxels_list)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=n_plots,
        subplot_titles=titles,
        specs=[[{'type': 'scatter3d'} for _ in range(n_plots)]],
        horizontal_spacing=0.05,
    )
    
    # Add each voxel grid
    for i, (voxels, title) in enumerate(zip(voxels_list, titles)):
        if isinstance(voxels, torch.Tensor):
            voxels = voxels.detach().cpu().numpy()
        
        # Get occupied coordinates
        occupied = voxels > threshold
        coords = np.argwhere(occupied)
        
        if len(coords) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=coords[:, 2],
                        colorscale='Viridis',
                        opacity=0.7,
                    ),
                    name=title,
                    showlegend=False,
                ),
                row=1, col=i+1
            )
    
    # Update layout
    fig.update_layout(
        title="Voxel Grid Comparison",
        height=600,
        width=300 * n_plots,
    )
    
    # Update all scenes to have same aspect ratio
    for i in range(n_plots):
        scene_name = f'scene{i+1}' if i > 0 else 'scene'
        fig.update_layout(**{
            scene_name: dict(
                aspectmode='cube',
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title=''),
            )
        })
    
    if save_path:
        fig.write_html(save_path)
    
    if show:
        fig.show()
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[Any, Any]:
    """
    Plot training history with loss and accuracy curves.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save plot
        show: Whether to display
        
    Returns:
        Figure and axes objects
    """
    # Lazy import matplotlib
    global plt
    if plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig, (ax1, ax2)


def create_class_distribution_plot(
    labels: Union[torch.Tensor, np.ndarray],
    class_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
) -> go.Figure:
    """
    Create interactive bar plot of class distribution.
    
    Args:
        labels: Class labels
        class_names: Names of classes
        save_path: Save path
        show: Whether to display
        
    Returns:
        Plotly figure
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Count occurrences
    unique, counts = np.unique(labels, return_counts=True)
    
    # Create bar plot
    fig = go.Figure(data=[
        go.Bar(
            x=[class_names[i] for i in unique],
            y=counts,
            text=counts,
            textposition='auto',
            marker_color='lightblue',
            marker_line_color='darkblue',
            marker_line_width=1.5,
        )
    ])
    
    # Update layout
    fig.update_layout(
        title="Class Distribution",
        xaxis_title="Class",
        yaxis_title="Count",
        showlegend=False,
        width=800,
        height=500,
    )
    
    if save_path:
        fig.write_html(save_path)
    
    if show:
        fig.show()
    
    return fig


def visualize_attention_maps(
    attention_maps: List[torch.Tensor],
    layer_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Any:
    """
    Visualize attention maps from different layers.
    
    Args:
        attention_maps: List of attention tensors
        layer_names: Names of layers
        save_path: Save path
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
    
    n_maps = len(attention_maps)
    fig, axes = plt.subplots(1, n_maps, figsize=(4 * n_maps, 4))
    
    if n_maps == 1:
        axes = [axes]
    
    for i, (att_map, ax) in enumerate(zip(attention_maps, axes)):
        # Take mean across channels and center slice
        if isinstance(att_map, torch.Tensor):
            att_map = att_map.detach().cpu().numpy()
        
        # Average across channels if needed
        if att_map.ndim == 5:  # B, C, D, H, W
            att_map = att_map[0].mean(axis=0)  # Take first sample, average channels
        elif att_map.ndim == 4:  # C, D, H, W
            att_map = att_map.mean(axis=0)
        
        # Take center slice
        center_slice = att_map[att_map.shape[0] // 2]
        
        # Plot
        im = ax.imshow(center_slice, cmap='hot', aspect='auto')
        ax.set_title(layer_names[i] if layer_names else f'Layer {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig 