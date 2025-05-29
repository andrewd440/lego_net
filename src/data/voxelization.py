"""Voxelization utilities for 3D data processing"""

import numpy as np
import torch
import trimesh
import open3d as o3d
from typing import Tuple, Optional, Union
from pathlib import Path


def mesh_to_voxels(
    mesh_path: Union[str, Path],
    voxel_size: int = 32,
    normalize: bool = True,
    padding: float = 0.1,
) -> torch.Tensor:
    """
    Convert a 3D mesh to a voxel grid.
    
    Args:
        mesh_path: Path to mesh file (OFF, OBJ, STL, etc.)
        voxel_size: Size of the voxel grid (will be cubic)
        normalize: Whether to normalize the mesh to unit cube
        padding: Padding factor for the bounding box
        
    Returns:
        Voxel grid tensor of shape (voxel_size, voxel_size, voxel_size)
    """
    # Load mesh using trimesh for better format support
    mesh = trimesh.load(mesh_path, force='mesh')
    
    if normalize:
        # Center mesh at origin
        mesh.vertices -= mesh.vertices.mean(axis=0)
        
        # Scale to unit cube with padding
        scale = (1.0 - 2 * padding) / np.max(np.abs(mesh.vertices))
        mesh.vertices *= scale
    
    # Convert to Open3D mesh for voxelization
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    
    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        o3d_mesh,
        voxel_size=2.0 / voxel_size  # Adjust for unit cube
    )
    
    # Convert to tensor
    voxels = voxel_grid_to_tensor(voxel_grid, voxel_size)
    
    return voxels


def voxelize_point_cloud(
    points: np.ndarray,
    voxel_size: int = 32,
    normalize: bool = True,
    padding: float = 0.1,
) -> torch.Tensor:
    """
    Convert a point cloud to a voxel grid.
    
    Args:
        points: Point cloud array of shape (N, 3)
        voxel_size: Size of the voxel grid
        normalize: Whether to normalize points to unit cube
        padding: Padding factor
        
    Returns:
        Voxel grid tensor
    """
    points = np.array(points)
    
    if normalize:
        # Center points
        points -= points.mean(axis=0)
        
        # Scale to unit cube with padding
        scale = (1.0 - 2 * padding) / np.max(np.abs(points))
        points *= scale
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Voxelize
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd,
        voxel_size=2.0 / voxel_size
    )
    
    return voxel_grid_to_tensor(voxel_grid, voxel_size)


def voxel_grid_to_tensor(
    voxel_grid: o3d.geometry.VoxelGrid,
    grid_size: int
) -> torch.Tensor:
    """
    Convert Open3D voxel grid to torch tensor.
    
    Args:
        voxel_grid: Open3D voxel grid
        grid_size: Target grid size
        
    Returns:
        Binary voxel tensor
    """
    # Initialize empty grid
    tensor = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.float32)
    
    # Get voxel indices
    voxels = voxel_grid.get_voxels()
    
    for voxel in voxels:
        # Convert voxel grid coordinates to tensor indices
        idx = voxel.grid_index
        # Map from voxel grid space to tensor space
        i = int((idx[0] + grid_size // 2) % grid_size)
        j = int((idx[1] + grid_size // 2) % grid_size)
        k = int((idx[2] + grid_size // 2) % grid_size)
        
        if 0 <= i < grid_size and 0 <= j < grid_size and 0 <= k < grid_size:
            tensor[i, j, k] = 1.0
    
    return tensor


def batch_voxelize(
    mesh_paths: list,
    voxel_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Voxelize multiple meshes in batch.
    
    Args:
        mesh_paths: List of mesh file paths
        voxel_size: Voxel grid size
        normalize: Whether to normalize meshes
        show_progress: Show progress bar
        
    Returns:
        Batch tensor of shape (N, voxel_size, voxel_size, voxel_size)
    """
    from tqdm import tqdm
    
    voxels_list = []
    
    iterator = tqdm(mesh_paths, desc="Voxelizing") if show_progress else mesh_paths
    
    for path in iterator:
        try:
            voxels = mesh_to_voxels(path, voxel_size, normalize)
            voxels_list.append(voxels)
        except Exception as e:
            print(f"Error voxelizing {path}: {e}")
            # Add empty voxel grid as fallback
            voxels_list.append(torch.zeros((voxel_size, voxel_size, voxel_size)))
    
    return torch.stack(voxels_list)


def compute_surface_normals(voxels: torch.Tensor) -> torch.Tensor:
    """
    Compute surface normals for voxel grid.
    
    Args:
        voxels: Binary voxel grid
        
    Returns:
        Normal vectors at surface voxels
    """
    # Compute gradients
    dx = torch.zeros_like(voxels)
    dy = torch.zeros_like(voxels)
    dz = torch.zeros_like(voxels)
    
    dx[1:-1, :, :] = voxels[2:, :, :] - voxels[:-2, :, :]
    dy[:, 1:-1, :] = voxels[:, 2:, :] - voxels[:, :-2, :]
    dz[:, :, 1:-1] = voxels[:, :, 2:] - voxels[:, :, :-2]
    
    # Normalize
    normals = torch.stack([dx, dy, dz], dim=-1)
    norm = torch.norm(normals, dim=-1, keepdim=True)
    normals = normals / (norm + 1e-8)
    
    return normals 