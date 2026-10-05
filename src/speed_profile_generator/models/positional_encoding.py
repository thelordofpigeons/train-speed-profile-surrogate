import torch
import torch.nn as nn
import numpy as np

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for normalized coordinates xi.
    
    Encodes scalar xi into a vector of size 2N using geometric frequencies.
    omega_i = base^(i/(N-1)) for i = 0..N-1
    output = [sin(omega_0 * xi), cos(omega_0 * xi), ..., sin(omega_{N-1} * xi), cos(omega_{N-1} * xi)]
    """
    def __init__(self, num_freqs: int = 48, base: float = 1000.0, debug: bool = False):
        super().__init__()
        self.num_freqs = num_freqs
        self.base = base
        
        # Frequencies are geometric from 1 to `base` (omega_i = base^(i/(N-1))),
        # so `base` is the highest angular frequency. A very large base makes
        # the encoding oscillate far faster than the track resolution, which
        # can add high-frequency texture to the output.
        if base > 1000.0 and debug:
            print(f"Warning: high PE base {base}; the highest frequency may be too fine for the track resolution.")

        i = torch.arange(num_freqs, dtype=torch.float32)
        omega = torch.pow(base, i / (num_freqs - 1))

        if debug:
             print(f"PosEnc: N={num_freqs}, Base={base}. Omega range: [{omega.min().item():.1f}, {omega.max().item():.1f}]")
        
        # Register as buffer so it's part of state_dict but not a parameter
        self.register_buffer("omega", omega)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xi (Tensor): [B, T, 1] normalized values in [0, 1].
            
        Returns:
            Tensor: [B, T, 2*N] encoded features.
        """
        # omega: [N] (from registered buffer)
        
        # xi: [B, T, 1]
        # Calculate product: xi * omega.view(1, 1, -1) -> [B, T, N]
        angle = xi * self.omega.view(1, 1, -1)
        
        prior_state = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1) # [B, T, 2N]
        return prior_state
