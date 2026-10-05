import torch

def compute_xi(x: torch.Tensor, x0: torch.Tensor, xf: torch.Tensor, eps: float = 1e-12, clamp: bool = True) -> torch.Tensor:
    """
    Compute normalized spatial coordinate xi = (x - x0) / (xf - x0).
    
    Args:
        x (Tensor): [B, T, 1] or [B, 1] physical positions.
        x0 (Tensor): [B, 1] start positions.
        xf (Tensor): [B, 1] end positions.
        eps (float): Small epsilon to prevent division by zero.
        clamp (bool): Whether to clamp xi to [0, 1].
        
    Returns:
        xi (Tensor): [B, T, 1] normalized positions.
    """
    # Ensure broadcasting: [B, 1] -> [B, 1, 1] if needed
    if x0.dim() == 2: x0 = x0.unsqueeze(1)
    if xf.dim() == 2: xf = xf.unsqueeze(1)
    if x.dim() == 2: x = x.unsqueeze(1)
        
    denominator = (xf - x0)
    xi = (x - x0) / (denominator + eps)
    
    if clamp:
        xi = torch.clamp(xi, 0.0, 1.0)
        
    return xi

def apply_hard_bc(xi: torch.Tensor, y: torch.Tensor, v0: torch.Tensor, vf: torch.Tensor) -> torch.Tensor:
    """
    Apply hard boundary conditions to the NN output.
    
    v(x) = (1 - xi) * v0 + xi * vf + xi * (1 - xi) * y
    
    Args:
        xi (Tensor): [B, T, 1] normalized positions.
        y (Tensor): [B, T, 1] NN core output.
        v0 (Tensor): [B, 1] start velocities.
        vf (Tensor): [B, 1] end velocities.
        
    Returns:
        v (Tensor): [B, T, 1] physical velocities strictly respecting BCs.
    """
    # Ensure broadcasting: [B, 1] -> [B, 1, 1]
    if v0.dim() == 2: v0 = v0.unsqueeze(1)
    if vf.dim() == 2: vf = vf.unsqueeze(1)
    
    # Term 1: Linear interpolation
    term_linear = (1.0 - xi) * v0 + xi * vf
    
    # Term 2: Correction
    term_correction = xi * (1.0 - xi) * y
    
    v = term_linear + term_correction
    return v
