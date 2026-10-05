import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Union, Any
from speed_profile_generator.utils.data_gen import generate_batch_dataset

class SpeedProfileDataset(Dataset):
    """
    Dataset wrapper for Speed Profile Generation (V2).
    
    Supports:
    - Mixture sampling by distance bins (controlled by config).
    - P-normalization (per-dimension).
    - Returns both normalized P (for model) and raw P (for physics/logging).
    """
    def __init__(self, 
                 num_samples: int, 
                 dist_range: tuple = (500.0, 10000.0),
                 seed: int = 42,
                 num_points: int = 200,
                 sampling_config: Optional[Dict[str, Any]] = None,
                 filter_regime: Optional[str] = None, # Legacy/Simple support
                 filter_dist_max: Optional[float] = None, # Simple support
                 filter_dist_min: Optional[float] = None, # Simple support
                 ):
        
        super().__init__()
        
        # Seed everything
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.data: List[Dict[str, torch.Tensor]] = []
        self.stats: Optional[Dict[str, torch.Tensor]] = None
        self.p_layout = ["mass", "v_max", "a_max", "b_max"] # 4 dims
        
        # Decide Strategy
        strategy = "simple"
        if sampling_config and sampling_config.get("strategy") == "mixture":
            strategy = "mixture"
            
        print(f"Initializing Dataset (V2). Strategy={strategy}, N={num_samples}, Seed={seed}")
        
        if strategy == "mixture":
            self._generate_mixture(num_samples, num_points, sampling_config)
        else:
            # Fallback to simple generation with optional filters
            # Map legacy filters to a single 'component' like config
            # Use the provided dist_range as base, but respect filter_dist_min/max if provided
            eff_min = filter_dist_min if filter_dist_min is not None else dist_range[0]
            eff_max = filter_dist_max if filter_dist_max is not None else dist_range[1]
            
            self._generate_simple(num_samples, num_points, eff_min, eff_max, filter_regime)

    def _generate_simple(self, num_samples: int, num_points: int, dist_min: float, dist_max: float, filter_regime: str):
        """Original generation logic, consolidated."""
        attempts = 0
        while len(self.data) < num_samples:
            remaining = num_samples - len(self.data)
            batch_size = min(remaining * 2 + 100, 2000)
            
            # Safety checks on ranges
            if dist_min >= dist_max:
                 dist_max = dist_min + 1000.0
            
            batch = generate_batch_dataset(
                batch_size=batch_size,
                dist_min=dist_min,
                dist_max=dist_max,
                num_points=num_points
            )
            
            for item in batch:
                if len(self.data) >= num_samples:
                    break
                    
                keep = True
                
                # Filter by Regime
                if filter_regime:
                    regime = item["regime"].item() # 0.0 or 1.0
                    if filter_regime == 'triangular' and regime != 0.0:
                        keep = False
                    elif filter_regime == 'trapezoidal' and regime != 1.0:
                        keep = False
                
                if keep:
                    self.data.append(item)
            
            attempts += 1
            if attempts > 100 and len(self.data) == 0:
                raise RuntimeError("Could not generate valid samples with current filters in simple mode.")

    def _generate_mixture(self, total_samples: int, num_points: int, config: Dict):
        """Generate data based on mixture components."""
        components = config.get("components", [])
        if not components:
            raise ValueError("Mixture strategy selected but no components provided.")
            
        # 1. Allocate counts
        # We need exact sum.
        counts = []
        weights = [c.get("weight", 1.0) for c in components]
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]
        
        current_sum = 0
        for i, w in enumerate(norm_weights):
            if i == len(components) - 1:
                # Last one takes remainder to ensure exact sum
                count = total_samples - current_sum
            else:
                count = int(round(w * total_samples))
            counts.append(count)
            current_sum += count
            
        print(f"Mixture Allocation: {[c['name'] for c in components]} -> {counts}")
        
        # 2. Generate per component
        for comp, count in zip(components, counts):
            if count == 0:
                continue
                
            dist_range = comp.get("dist_range", [500.0, 10000.0])
            regime = comp.get("regime", None) # Optional, strictly speaking just dist bins prefered
            
            print(f"  - Generating {count} samples for '{comp.get('name')}' (Range: {dist_range}, Regime: {regime})")
            
            # Use temporary list to hold this component's data
            comp_data = []
            attempts = 0
            
            d_min, d_max = dist_range[0], dist_range[1]
            
            while len(comp_data) < count:
                remaining = count - len(comp_data)
                batch_size = min(remaining * 2 + 100, 2000)
                
                batch = generate_batch_dataset(
                    batch_size=batch_size,
                    dist_min=d_min,
                    dist_max=d_max,
                    num_points=num_points
                )
                
                for item in batch:
                    if len(comp_data) >= count:
                        break
                    
                    keep = True
                    if regime:
                        r_val = item["regime"].item()
                        if regime == 'triangular' and r_val != 0.0: keep = False
                        elif regime == 'trapezoidal' and r_val != 1.0: keep = False
                    
                    if keep:
                        comp_data.append(item)
                        
                attempts += 1
                if attempts > 200 and len(comp_data) == 0: 
                    # Warning instead of hard crash maybe? No, let's crash to be safe.
                    raise RuntimeError(f"Could not generate samples for component {comp.get('name')}")
            
            self.data.extend(comp_data)
            
    def compute_stats(self) -> Dict[str, Any]:
        """
        Compute mean and std of P across the entire dataset.
        Returns dict suitable for saving to JSON.
        """
        if not self.data:
            return {}
            
        # Stack all P vectors: [N, 4]
        all_P = torch.stack([d['P'] for d in self.data], dim=0) # [N, 4]
        
        mean = torch.mean(all_P, dim=0) # [4]
        std = torch.std(all_P, dim=0)   # [4]
        
        # Avoid div by zero (unlikely for these physical params but good practice)
        eps = 1e-8
        std = torch.clamp(std, min=eps)
        
        stats = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "eps": eps,
            "P_layout": self.p_layout
        }
        return stats
        
    def set_stats(self, stats: Dict[str, Any]):
        """
        Set normalization statistics.
        """
        if not stats:
            self.stats = None
            return
            
        self.stats = {
            "mean": torch.tensor(stats["mean"], dtype=torch.float32),
            "std": torch.tensor(stats["std"], dtype=torch.float32),
            "eps": stats.get("eps", 1e-8)
        }
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # item['P'] is currently raw. 
        # We need to return {'P': normalized_if_active, 'P_raw': raw}
        
        P_raw = item['P'].clone() # [4]
        
        if self.stats is not None:
            mean = self.stats["mean"]#.to(P_raw.device) # Assume CPU for dataset
            std = self.stats["std"]#.to(P_raw.device)
            P_norm = (P_raw - mean) / std
        else:
            P_norm = P_raw.clone()
            
        # Return new dict to avoid mutating stored data in memory
        new_item = item.copy()
        new_item['P'] = P_norm
        new_item['P_raw'] = P_raw
        
        return new_item
