import pytest
import torch
import numpy as np
import os
from speed_profile_generator.utils.preprocessor import PPreprocessor

def test_preprocessor_cont_fit_transform(tmp_path):
    # 1. Generate realistic dummy data
    np.random.seed(42)
    N = 1000
    L = np.random.uniform(500, 100000, N)        # meters: 500m to 100km
    v0 = np.random.uniform(0, 30, N)             # m/s
    vf = np.random.uniform(0, 30, N)             # m/s
    dur = np.random.uniform(30, 10000, N)        # seconds
    M = np.random.uniform(50000, 1000000, N)     # kg
    v_seg_max = np.random.uniform(15, 45, N)     # m/s
    
    # Introduce some heavy-tail outliers
    dur[0:10] = 50000  
    M[0:10] = 5000000
    
    p_cont_list = np.stack([L, v0, vf, dur, M, v_seg_max], axis=1).astype(np.float32)
    
    # 2. Fit
    prep = PPreprocessor()
    stats = prep.fit_cont(list(p_cont_list), use_robust_stats=True)
    
    assert stats["continuous_schema"] is True
    assert prep.mean is not None
    assert prep.std is not None
    assert prep.clip_lo is not None
    
    # 3. Transform
    P_tensor_orig = torch.tensor(p_cont_list, dtype=torch.float32)
    P_tensor = P_tensor_orig.clone()
    
    P_norm1 = prep.transform_cont(P_tensor)
    
    # Check no NaNs
    assert not torch.isnan(P_norm1).any(), "Found NaNs in normalized output"
    
    # Check bounds (roughly standard normal since we clipped extreme outliers)
    mean_out = P_norm1.mean(dim=0)
    std_out = P_norm1.std(dim=0)
    
    assert torch.allclose(mean_out, torch.zeros_like(mean_out), atol=0.1), f"Mean not centered: {mean_out}"
    assert torch.allclose(std_out, torch.ones_like(std_out), atol=0.2), f"Std not scaled: {std_out}"
    
    # 4. Reproducibility & No-mutation
    # Re-transforming the *original* clone should be bitwise identical
    P_tensor_orig_clone = P_tensor_orig.clone()
    P_norm2 = prep.transform_cont(P_tensor_orig_clone)
    assert torch.equal(P_norm1, P_norm2), "Transform is not deterministic!"
    
    # Ensure source tensor was NOT mutated
    assert torch.equal(P_tensor_orig, P_tensor_orig_clone), "Source tensor mutated!"
    
    # 5. Save Load
    stats_path = str(tmp_path / "test_prep_stats.json")
    prep.save(stats_path)
    
    prep2 = PPreprocessor()
    prep2.load(stats_path)
    
    P_norm3 = prep2.transform_cont(P_tensor_orig.clone())
    assert torch.equal(P_norm1, P_norm3), "Save/Load broke reproducibility!"

