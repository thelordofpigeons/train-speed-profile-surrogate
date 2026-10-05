import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across multiple libraries.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")
