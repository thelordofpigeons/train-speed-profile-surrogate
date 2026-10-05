import torch
import torch.nn as nn
import torch.nn.functional as F

class ModulatedResidualBlock(nn.Module):
    """
    Modulated Residual Block.
    
    h_out = A(h_in) * B(P) + h_in
    
    A: Linear(hidden_dim -> hidden_dim) (plus activation)
    B: Linear(param_dim -> hidden_dim) (projection of parameters)
    """
    def __init__(self, hidden_dim: int, param_dim: int, activation=nn.SiLU()):
        super().__init__()
        
        self.layer_A = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation
        )
        
        # B projects parameters P to modulation weights
        self.layer_B = nn.Linear(param_dim, hidden_dim)
        
        # Initialize B to be close to 1 or small modifications? 
        # Usually zero-init for modulation main branch is good, but here it's multiplicative.
        # If B(P) is near 0, then block starts as identity (if we did A*B + h).
        # But if B(P) is modulation, maybe we want it near 1?
        # User constraint: h_out = A(h_in) * B(P) + h_in
        # If B(P) is 0, output is h_in. If B(P) is 1, output is A(h) + h.
        # Standard FiLM is usually scale/shift. Here it's pure gating?
        # Contract: h_out = A(h_in) * B(P) + h_in.
        pass

    def forward(self, h_in: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_in: [B, T, H]
            P: [B, dP]
            
        Returns:
            h_out: [B, T, H]
        """
        # A(h_in)
        a_out = self.layer_A(h_in) # [B, T, H]
        
        # B(P)
        # P is [B, dP]. Project to [B, H].
        b_out = self.layer_B(P) # [B, H]
        
        # Broadcast B over T
        b_out = b_out.unsqueeze(1) # [B, 1, H]
        
        # Modulate
        modulated = a_out * b_out # [B, T, H]
        
        # Residual
        return modulated + h_in
