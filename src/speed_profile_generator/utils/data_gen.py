import numpy as np
import torch
from typing import Dict, Tuple

def generate_trajectory_kinematic(
    x0: float, xf: float, v0: float, vf: float,
    v_max: float, a_max: float, b_max: float,
    num_points: int = 100
) -> Dict[str, torch.Tensor]:
    """
    Generates a deterministic kinematic trajectory (Accel-Coast-Brake).
    
    Args:
        x0, xf: Start and end positions (m).
        v0, vf: Start and end velocities (m/s).
        v_max: Maximum allowed speed (m/s).
        a_max: Maximum acceleration (m/s^2) > 0.
        b_max: Maximum deceleration (m/s^2) > 0.
        num_points: Number of points in the spatial grid.
        
    Returns:
        Dict with keys:
            'x': [T, 1]
            'v_ref': [T, 1]
            't': [T, 1] (optional, time profile)
    """
    total_dist = xf - x0
    assert total_dist > 0, f"xf ({xf}) must be > x0 ({x0})"
    
    # 1. Theoretical distances for full acceleration to v_max
    # v^2 = u^2 + 2as -> s = (v^2 - u^2) / 2a
    
    # Check if v0 > v_max (should not handle, but clamp or error)
    v0 = min(v0, v_max)
    vf = min(vf, v_max)
    
    # Distance to accel: v0 -> v_max
    dist_acc = (v_max**2 - v0**2) / (2 * a_max)
    
    # Distance to decel: v_max -> vf
    dist_dec = (v_max**2 - vf**2) / (2 * b_max)
    
    # 2. Determine Profile Type
    if total_dist >= dist_acc + dist_dec:
        # Trapezoidal Profile
        v_peak = v_max
        regime = 1.0 # 1=Trapezoidal
        # Phase switch points (relative to x0)
        x1 = x0 + dist_acc
        x2 = xf - dist_dec # Start braking
    else:
        # Triangular Profile (Peak speed < v_max)
        regime = 0.0 # 0=Triangular
        # Solve for v_peak:
        # (v_peak^2 - v0^2)/2a + (v_peak^2 - vf^2)/2b = total_dist
        # v_peak^2 * (1/2a + 1/2b) = total_dist + v0^2/2a + vf^2/2b
        # Let K = (1/2a + 1/2b)
        # v_peak^2 * K = RHS
        
        K = (1.0 / (2 * a_max)) + (1.0 / (2 * b_max))
        RHS = total_dist + (v0**2 / (2 * a_max)) + (vf**2 / (2 * b_max))
        v_peak_sq = RHS / K
        v_peak = np.sqrt(v_peak_sq)
        
        # New switch points
        dist_acc_tri = (v_peak**2 - v0**2) / (2 * a_max)
        x1 = x0 + dist_acc_tri
        x2 = x1 # No coasting
        
    # 3. Generate Grid
    x_grid = np.linspace(x0, xf, num_points)
    
    # 4. Compute Velocity Profile vectorized
    v_grid = np.zeros_like(x_grid)
    
    # Phase 1: Accel (x0 <= x <= x1)
    # v(x) = sqrt(v0^2 + 2 * a * (x - x0))
    mask_acc = (x_grid <= x1)
    v_grid[mask_acc] = np.sqrt(v0**2 + 2 * a_max * (x_grid[mask_acc] - x0))
    
    # Phase 2: Coast (x1 < x < x2)
    if x2 > x1:
        mask_coast = (x_grid > x1) & (x_grid < x2)
        v_grid[mask_coast] = v_peak # Constant speed
        
    # Phase 3: Decel (x >= x2)
    # v(x) = sqrt(v_peak^2 - 2 * b * (x - x2)) -- Wait, v^2 = u^2 + 2as. a is negative.
    # v^2 = v_peak^2 - 2 * b * (x - x2)
    mask_dec = (x_grid >= x2)
    # Numerical safety: abs to avoid negative close to zero
    v_sq_dec = v_peak**2 - 2 * b_max * (x_grid[mask_dec] - x2)
    v_sq_dec = np.maximum(v_sq_dec, 0.0) # Clamp negative
    v_grid[mask_dec] = np.sqrt(v_sq_dec)
    
    # Ensure exact endpoints
    v_grid[0] = v0
    v_grid[-1] = vf
    
    # Convert to Tensors
    return {
        "x": torch.tensor(x_grid, dtype=torch.float32).unsqueeze(-1), # [T, 1]
        "v_ref": torch.tensor(v_grid, dtype=torch.float32).unsqueeze(-1), # [T, 1]
        "regime": torch.tensor([regime], dtype=torch.float32) # [1]
    }

def generate_batch_dataset(
    batch_size: int,
    dist_min: float = 500.0, dist_max: float = 10000.0,
    speed_limit_range: Tuple[float, float] = (20.0, 80.0),
    accel_range: Tuple[float, float] = (0.5, 1.5),
    num_points: int = 100
) -> Dict[str, torch.Tensor]:
    """
    Generates a batch of synthetic data.
    """
    data_list = []
    
    for _ in range(batch_size):
        # Sample parameters
        delta_x = np.random.uniform(dist_min, dist_max)
        x0 = 0.0 # Can be random, but relative matters most
        xf = x0 + delta_x
        
        v_max = np.random.uniform(*speed_limit_range)
        a_max = np.random.uniform(*accel_range)
        b_max = np.random.uniform(*accel_range)
        
        # Sample Mass (New in V2)
        mass = np.random.uniform(70000.0, 500000.0)
        
        # Sample BCs
        v0 = np.random.uniform(0, v_max * 0.8) # bias slightly lower than max to allow accel
        vf = np.random.uniform(0, v_max * 0.8)
        
        traj = generate_trajectory_kinematic(x0, xf, v0, vf, v_max, a_max, b_max, num_points)
        
        # Pack Parameters P
        # P: [mass, v_max, a_max, b_max]
        P = torch.tensor([mass, v_max, a_max, b_max], dtype=torch.float32)
        
        item = {
            "x": traj["x"],      # [T, 1]
            "v_ref": traj["v_ref"],  # [T, 1]
            "P": P,              # [4]
            "x0": torch.tensor([x0], dtype=torch.float32),
            "xf": torch.tensor([xf], dtype=torch.float32),
            "v0": torch.tensor([v0], dtype=torch.float32),
            "vf": torch.tensor([vf], dtype=torch.float32),
            "regime": traj["regime"], # [1]
        }
        data_list.append(item)
        
    return data_list
