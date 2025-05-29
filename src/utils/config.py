"""Configuration management utilities"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    _validate_config(config)
    
    # Expand paths
    config = _expand_paths(config)
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['model', 'data', 'training', 'optimizer']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate model parameters
    if config['model']['num_classes'] <= 0:
        raise ValueError("num_classes must be positive")
    
    if not all(dim > 0 for dim in config['model']['input_shape']):
        raise ValueError("All input dimensions must be positive")
    
    # Validate training parameters
    if config['training']['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
    
    if config['training']['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")
    
    # Validate device
    valid_devices = ['cpu', 'cuda', 'mps']
    if config['training']['device'] not in valid_devices:
        raise ValueError(f"Invalid device. Must be one of: {valid_devices}")


def _expand_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand relative paths in configuration to absolute paths.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with expanded paths
    """
    # Paths to expand
    path_keys = [
        ('data', 'root_dir'),
        ('data', 'cache_dir'),
        ('training', 'checkpoint_dir'),
        ('training', 'log_dir'),
        ('visualization', 'plot_dir'),
    ]
    
    for section, key in path_keys:
        if section in config and key in config[section]:
            path = Path(config[section][key])
            if not path.is_absolute():
                config[section][key] = str(path.absolute())
    
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    import copy
    merged = copy.deepcopy(base_config)
    
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    return deep_update(merged, override_config) 