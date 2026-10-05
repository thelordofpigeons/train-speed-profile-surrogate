import torch
import torch.nn as nn

class SpeedProfileLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 0.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.mse = nn.MSELoss()
        
    def forward(self, v_pred: torch.Tensor, v_ref: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_pred: [B, T, 1]
            v_ref: [B, T, 1]
            x: [B, T, 1] (needed for smoothness)
        """
        # Primary Loss
        loss_mse = self.mse(v_pred, v_ref)
        
        loss_reg = 0.0
        if self.lambda_smooth > 0.0:
            # Smoothness: (dv/dx)^2
            # dx shape: [B, T-1, 1]
            dx = x[:, 1:, :] - x[:, :-1, :]
            dv = v_pred[:, 1:, :] - v_pred[:, :-1, :]
            
            # Avoid div by zero (should not happen in grid, but eps is safe)
            grad = dv / (dx + 1e-6)
            
            # Penalize mean square gradient? Or 2nd derivative?
            # User request: "compute it as (dv/dx)^2"
            loss_reg = torch.mean(grad**2)
            
        return loss_mse + self.lambda_smooth * loss_reg, loss_mse, loss_reg
