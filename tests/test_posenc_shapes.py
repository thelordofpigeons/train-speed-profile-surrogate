import pytest
import torch
import math
from speed_profile_generator.models.positional_encoding import SinusoidalPositionalEncoding

def test_posenc_shapes_and_values():
    # Setup standard config from Titan RTX plan
    num_freqs = 48
    base = 1000.0
    
    posenc = SinusoidalPositionalEncoding(num_freqs=num_freqs, base=base, debug=False)
    
    # 1. Check shape
    B, T = 4, 100
    xi = torch.linspace(0.0, 1.0, T).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1) # [B, T, 1]
    
    out = posenc(xi)
    
    expected_dim = 2 * num_freqs
    assert out.shape == (B, T, expected_dim), f"Expected shape {(B, T, expected_dim)}, got {out.shape}"
    
    # 2. NaN Checks over full range
    assert not torch.isnan(out).any(), "Found NaNs in positional encoding output!"
    assert not torch.isinf(out).any(), "Found Infs in positional encoding output!"
    
    # 3. Value Check at xi=0
    # At xi=0, sin(0)=0, cos(0)=1
    xi_0 = torch.zeros((1, 1, 1))
    out_0 = posenc(xi_0)
    
    sins = out_0[0, 0, :num_freqs]
    coss = out_0[0, 0, num_freqs:]
    
    assert torch.allclose(sins, torch.zeros_like(sins), atol=1e-6)
    assert torch.allclose(coss, torch.ones_like(coss), atol=1e-6)

def test_speed_profile_generator_config_fail():
    from speed_profile_generator.models.speed_profile_generator import SpeedProfileGenerator
    
    with pytest.raises(ValueError, match="requires explicit `num_freqs` and `freq_base`"):
        SpeedProfileGenerator(param_dim=8, num_freqs=None)
