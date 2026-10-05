import pytest
import torch
from speed_profile_generator.models.speed_profile_generator import SpeedProfileGenerator

def test_system_integration():
    B, T = 2, 10
    param_dim = 6
    model = SpeedProfileGenerator(param_dim=param_dim, num_freqs=16, freq_base=10.0)

    x = torch.linspace(0, 100, T).view(1, T, 1).repeat(B, 1, 1)
    P_feats = torch.randn(B, param_dim)

    # P_raw layout: [x0, xf, v0, vf, st_start, st_end, dur, M]
    P_raw = torch.zeros(B, 8)
    P_raw[:, 1] = 100.0
    P_raw[:, 2] = 5.0
    P_raw[:, 3] = 15.0
    v0 = P_raw[:, 2:3]
    vf = P_raw[:, 3:4]

    v = model.forward_x(x, P_feats, P_raw)

    # Check shapes
    assert v.shape == (B, T, 1)

    # Check BC enforcement exactly at endpoints
    # First point
    assert torch.allclose(v[:, 0, :], v0, atol=1e-5)
    # Last point
    assert torch.allclose(v[:, -1, :], vf, atol=1e-5)

    # Check gradient flow (simple check)
    v.sum().backward()
    assert model.core.output_proj.weight.grad is not None

def test_legacy_forward_is_removed():
    model = SpeedProfileGenerator(param_dim=6, num_freqs=16, freq_base=10.0)
    with pytest.raises(NotImplementedError):
        model(torch.zeros(1, 4, 1), torch.zeros(1, 6))

def test_positional_encoding_shape():
    from speed_profile_generator.models.positional_encoding import SinusoidalPositionalEncoding
    N = 16
    pe = SinusoidalPositionalEncoding(num_freqs=N, base=10.0, debug=False)
    xi = torch.rand(2, 10, 1)
    out = pe(xi)
    assert out.shape == (2, 10, 2*N)
