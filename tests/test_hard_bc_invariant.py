import pytest
import torch
from speed_profile_generator.models.speed_profile_generator import SpeedProfileGenerator

def test_hard_bc_invariant():
    """
    Test that the neural network's boundary condition enforcement
    guarantees exactly v0 at xi=0 and vf at xi=1, regardless of weights.
    We test this with strictly random parameters and raw bounds.
    """
    # 1. Init model with complete random weights
    model = SpeedProfileGenerator(
        param_dim=8,  # (P_cont=6) + (2 emb_dim=2*4=8 ? wait, dim is explicit)
        num_freqs=16,
        freq_base=10.0,
        emb_dim='auto', 
        num_stations=50,
        track_feat_dim=0,
        extra_features_dim=0
    )
    
    # Random input ranges
    B, T = 8, 1  # Testing isolated points
    
    xi_0 = torch.zeros((B, T, 1))
    xi_1 = torch.ones((B, T, 1))
    
    # Dummy Params
    v0_true = torch.rand((B, 1)) * 30.0
    vf_true = torch.rand((B, 1)) * 30.0
    
    # We construct P_raw with these at index 2 and 3
    P_raw = torch.zeros((B, 8))
    P_raw[:, 2] = v0_true.squeeze()
    P_raw[:, 3] = vf_true.squeeze()
    
    # P_cont expects 6 continuous normalized features
    # Doesn't matter what they are since we expect BC wrapper to overpower them
    P_cont = torch.randn((B, 6))
    
    st_ids = torch.randint(0, 50, (B, 2), dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        v_hat_0 = model.forward_xi(xi_0, P_cont, P_raw, station_ids=st_ids)
        v_hat_1 = model.forward_xi(xi_1, P_cont, P_raw, station_ids=st_ids)
        
    v_hat_0 = v_hat_0.view(B, 1)
    v_hat_1 = v_hat_1.view(B, 1)
        
    diff_0 = torch.abs(v_hat_0 - v0_true)
    diff_1 = torch.abs(v_hat_1 - vf_true)
    
    assert torch.all(diff_0 < 1e-5), f"BC failed at xi=0: Expected {v0_true}, got {v_hat_0}"
    assert torch.all(diff_1 < 1e-5), f"BC failed at xi=1: Expected {vf_true}, got {v_hat_1}"
