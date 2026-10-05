import pytest
import torch
from speed_profile_generator.utils.bc import compute_xi, apply_hard_bc

def test_bc_functions():
    # Test compute_xi
    x = torch.tensor([[[0.0], [5.0], [10.0]]])
    x0 = torch.tensor([[0.0]])
    xf = torch.tensor([[10.0]])
    xi = compute_xi(x, x0, xf)
    assert torch.allclose(xi, torch.tensor([[[0.0], [0.5], [1.0]]]))
    
    # Test apply_hard_bc
    y = torch.tensor([[[0.1], [0.2], [-0.1]]]) # Arbitrary NN output
    v0 = torch.tensor([[10.0]])
    vf = torch.tensor([[20.0]])
    
    # At x=x0 (xi=0): v = v0
    # At x=xf (xi=1): v = vf
    # At x=mid (xi=0.5): v = 0.5*v0 + 0.5*vf + 0.25*y
    
    xi_test = torch.tensor([[[0.0], [0.5], [1.0]]])
    v = apply_hard_bc(xi_test, y, v0, vf)
    
    assert torch.allclose(v[0, 0, 0], v0)
    assert torch.allclose(v[0, 2, 0], vf)
    
    expected_mid = 0.5 * 10.0 + 0.5 * 20.0 + 0.25 * 0.2
    assert torch.allclose(v[0, 1, 0], torch.tensor(expected_mid))
