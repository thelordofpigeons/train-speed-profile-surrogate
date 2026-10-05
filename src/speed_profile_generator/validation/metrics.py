import torch
import numpy as np
from typing import Dict, Tuple

def compute_bc_errors(v_pred: torch.Tensor, v0: torch.Tensor, vf: torch.Tensor) -> Dict[str, float]:
    """
    Computes max absolute errors at endpoints.
    Args:
        v_pred: [B, T, 1]
        v0: [B, 1]
        vf: [B, 1]
    """
    # Start error
    # v_pred[:, 0, :] shape [B, 1]
    err_start = torch.abs(v_pred[:, 0, :] - v0)
    max_err_start = torch.max(err_start).item()
    
    # End error
    err_end = torch.abs(v_pred[:, -1, :] - vf)
    max_err_end = torch.max(err_end).item()
    
    return {
        "max_bc_error_start": max_err_start,
        "max_bc_error_end": max_err_end
    }

def check_violations(v_pred: torch.Tensor, P: torch.Tensor) -> Dict[str, float]:
    """
    Checks non-negativity and speed limit violations.
    Args:
        v_pred: [B, T, 1]
        P: [B, dP] where P[:, 1] is v_max
    """
    # 1. Non-negativity
    # Point-level
    min_val = torch.min(v_pred).item()
    count_neg = torch.sum(v_pred < -1e-6).item()
    
    # Trajectory-level: count trajectories that have ANY violation
    # any over dim T (dim 1)
    # (v_pred < -1e-6) is [B, T, 1]
    # any(dim=1) -> [B, 1]
    traj_violation_mask = (v_pred < -1e-6).any(dim=1).any(dim=-1)
    frac_neg_traj = traj_violation_mask.float().mean().item()
    
    # 2. Speed Limit
    # v_max is P[:, 1] -> [B]
    v_max = P[:, 1].view(-1, 1, 1) # Broadcast to [B, 1, 1]
    
    # Excess: v - v_max
    excess = v_pred - v_max
    max_excess = torch.max(excess).item()
    
    # Trajectory-level violations (using a small tolerance?)
    # "fraction of trajectories violating" - let's use tol=1e-4
    tol = 1e-4
    viol_mask = (excess > tol)
    count_speed_viol_points = torch.sum(viol_mask).item()
    
    traj_speed_viol_mask = viol_mask.any(dim=1).any(dim=-1)
    frac_speed_viol_traj = traj_speed_viol_mask.float().mean().item()
    
    return {
        "min_velocity": min_val,
        "count_neg_points": count_neg,
        "frac_neg_traj": frac_neg_traj,
        "max_speed_excess": max_excess,
        "frac_speed_viol_traj": frac_speed_viol_traj
    }

def compute_regression_metrics(v_pred: torch.Tensor, v_ref: torch.Tensor) -> Dict[str, float]:
    """
    Computes RMSE and MaxAbsError for All points and Interior points (1:-1).
    Args:
        v_pred: [B, T, 1]
        v_ref: [B, T, 1]
    """
    # ALL
    diff = v_pred - v_ref
    mse_all = torch.mean(diff**2).item()
    rmse_all = np.sqrt(mse_all)
    max_abs_all = torch.max(torch.abs(diff)).item()
    
    # INTERIOR
    if v_pred.shape[1] > 2:
        diff_int = v_pred[:, 1:-1, :] - v_ref[:, 1:-1, :]
        mse_int = torch.mean(diff_int**2).item()
        rmse_int = np.sqrt(mse_int)
        max_abs_int = torch.max(torch.abs(diff_int)).item()
    else:
        rmse_int = rmse_all
        max_abs_int = max_abs_all
        
    return {
        "RMSE_all": rmse_all,
        "RMSE_interior": rmse_int,
        "MaxAbsError_all": max_abs_all,
        "MaxAbsError_interior": max_abs_int
    }
