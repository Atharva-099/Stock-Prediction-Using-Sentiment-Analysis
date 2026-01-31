"""
Utility functions for reproducibility and common operations.
"""
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): Random seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Optional: These can slow down training but ensure full reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
