
import torch
import json
import os
import numpy as np
from typing import Dict, List, Optional, Union

class PPreprocessor:
    """
    Preprocessor for Parameter Vector P.
    
    Features:
    - Fits mean/std on training data (feature-wise).
    - Ignores NaNs during fit (though P should be clean now).
    - Transform: Normalizes P using fit stats.
        
    P_raw structure (8 dims):
    0: x0
    1: xf
    2: v0
    3: vf
    4: noStartSt
    5: noEndSt
    6: duration
    7: M
    
    Output P_feats structure (8 dims):
    Normalized P_raw.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self.eps = 1e-6
        
    def fit(self, dataset, num_samples: int = 10000) -> Dict[str, List[float]]:
        """
        Compute mean/std from dataset.
        """
        print(f"Fitting PPreprocessor on {len(dataset)} samples...")
        
        # Collect Ps
        P_list = []
        indices = np.random.choice(len(dataset), size=min(len(dataset), num_samples), replace=False)
        
        for idx in indices:
            # dataset[idx] returns dict
            item = dataset[idx]
            P = item['P'] # Tensor or numpy
            if isinstance(P, torch.Tensor):
                P_list.append(P.numpy())
            else:
                P_list.append(P)
            
        P_arr = np.stack(P_list, axis=0) # [N, 8]
        
        # Compute stats ignoring NaNs (just in case)
        with np.errstate(invalid='ignore', divide='ignore'):
            self.mean = np.nanmean(P_arr, axis=0)
            self.std = np.nanstd(P_arr, axis=0)
            
        # Handle all-NaN columns: set mean=0.0, std=1.0
        nan_cols = np.isnan(self.mean)
        if np.any(nan_cols):
             print(f"Warning: Found all-NaN columns at indices {np.where(nan_cols)[0]}. Handling robustly (mean=0, std=1).")
             self.mean[nan_cols] = 0.0
             self.std[nan_cols] = 1.0
        
        # Avoid division by zero
        self.std[self.std < self.eps] = 1.0
        
        stats = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist()
        }
        return stats
        
    def transform(self, P: torch.Tensor) -> torch.Tensor:
        """
        Args:
            P: [B, 8] raw parameters
            
        Returns:
            P_feats: [B, 8] normalized (NaNs filled with 0.0 post-normalization or mean pre-normalization)
        """
        device = P.device
        dtype = P.dtype
        
        if self.mean is None:
            raise ValueError("PPreprocessor must be fitted explicitly or loaded before transform.")
            
        mean = torch.tensor(self.mean, device=device, dtype=dtype)
        std = torch.tensor(self.std, device=device, dtype=dtype)
        
        # Robustness: Fill NaNs in Input P with Mean BEFORE calc? 
        # Or standard z-score algebra: (NaN - mu)/std -> NaN. Then fill NaN with 0.
        # 0.0 in z-score space = mean in original space.
        # So (P - mean)/std, then nan_to_num(0.0) IS imputing with mean.
        
        # Normalize
        P_norm = (P - mean) / std
        
        # Fill NaNs with 0.0 (equivalent to imputing with training mean)
        P_norm = torch.nan_to_num(P_norm, nan=0.0)
        
        return P_norm
        
    def fit_cont(self, p_cont_list: List[np.ndarray], use_robust_stats: bool = True) -> Dict[str, List[float]]:
        """
        New Interface: Fits exclusively on continuous vectors [L, v0, vf, dur, M, v_seg_max].
        Args:
            p_cont_list: List of [N, 6] arrays or analogous
            use_robust_stats: Whether to compute p1/p99 clip bounds based on *transformed* features.
        """
        print(f"Fitting PPreprocessor (Cont Schema) on {len(p_cont_list)} raw samples...")
        
        P_arr = np.array(p_cont_list, dtype=np.float32) # [N, 6]
        assert P_arr.shape[1] == 6, f"Expected 6 continuous dimensions, got {P_arr.shape[1]}"
        
        # Guard assertions
        assert np.all(P_arr[:, 0] > 0), "Found L <= 0 during PPreprocessor.fit_cont"
        assert np.all(P_arr[:, 4] > 0), "Found M <= 0 during PPreprocessor.fit_cont"
        
        # 1. Transform raw to space
        P_t = P_arr.copy()
        P_t[:, 3] = np.log1p(P_t[:, 3]) # dur -> log1p(dur)
        P_t[:, 4] = np.log(P_t[:, 4])   # M -> log(M)
        
        self.use_robust_stats = use_robust_stats
        
        # 2. Compute clip bounds
        if use_robust_stats:
            p1 = np.percentile(P_t, 1, axis=0)
            p99 = np.percentile(P_t, 99, axis=0)
            
            # Clip BEFORE computing mean/std so outliers don't pull the mean
            P_t_clipped = np.clip(P_t, p1, p99)
            
            self.clip_lo = p1.tolist()
            self.clip_hi = p99.tolist()
        else:
            P_t_clipped = P_t
            self.clip_lo = None
            self.clip_hi = None
            
        # 3. Compute mean/std
        with np.errstate(invalid='ignore', divide='ignore'):
            self.mean = np.nanmean(P_t_clipped, axis=0)
            self.std = np.nanstd(P_t_clipped, axis=0)
            
        nan_cols = np.isnan(self.mean)
        if np.any(nan_cols):
             self.mean[nan_cols] = 0.0
             self.std[nan_cols] = 1.0
             
        self.std[self.std < self.eps] = 1.0
        
        stats = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "clip_lo": self.clip_lo,
            "clip_hi": self.clip_hi,
            "continuous_schema": True
        }
        return stats
        
    def transform_cont(self, P_cont: torch.Tensor) -> torch.Tensor:
        """
        Transform [B, 6] continuous vector.
        Logs applied exactly once here, followed by clip and standard scale.
        """
        device = P_cont.device
        dtype = P_cont.dtype
        
        if self.mean is None:
            raise ValueError("PPreprocessor not fitted.")
            
        P_t = P_cont.clone()
        
        # 1. Apply log transforms FIRST
        # dur_t
        P_t[:, 3] = torch.log1p(P_t[:, 3].clamp(min=0.0))
        # M_t
        P_t[:, 4] = torch.log(P_t[:, 4].clamp(min=1e-6))
        
        # 2. Apply clip bounds (robust scale)
        if getattr(self, 'use_robust_stats', False) and self.clip_lo is not None and self.clip_hi is not None:
            # clip_lo/hi are lists of 6 floats
            lower = torch.tensor(self.clip_lo, device=device, dtype=dtype)
            upper = torch.tensor(self.clip_hi, device=device, dtype=dtype)
            P_t = torch.maximum(torch.minimum(P_t, upper), lower)
            
        # 3. Z-score
        mean = torch.tensor(self.mean, device=device, dtype=dtype)
        std = torch.tensor(self.std, device=device, dtype=dtype)
        
        P_norm = (P_t - mean) / std
        P_norm = torch.nan_to_num(P_norm, nan=0.0)
        
        return P_norm
        
    def save(self, path: str):
        stats = {
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else [],
            "clip_lo": getattr(self, 'clip_lo', None),
            "clip_hi": getattr(self, 'clip_hi', None),
            "use_robust_stats": getattr(self, 'use_robust_stats', False)
        }
        with open(path, 'w') as f:
            json.dump(stats, f, indent=4)
            
    def load(self, path: str):
        with open(path, 'r') as f:
            stats = json.load(f)
        self.mean = np.array(stats['mean'], dtype=np.float32) if len(stats['mean']) > 0 else None
        self.std = np.array(stats['std'], dtype=np.float32) if len(stats.get('std', [])) > 0 else None
        
        if 'clip_lo' in stats and stats['clip_lo'] is not None:
            self.clip_lo = stats['clip_lo']
            self.clip_hi = stats['clip_hi']
            self.use_robust_stats = stats.get('use_robust_stats', True)
        else:
            self.clip_lo = None
            self.clip_hi = None
            self.use_robust_stats = False
