
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List

class SpeedProfileLossV3(nn.Module):
    """
    V3 Loss Module for Hybrid Training.
    
    Objectives:
    1. Peak Speed Loss: Match vmax.
    2. Curvature Loss: Smoothness via discrete second difference (STRICT: no dxi division).
    3. Cruise Loss: Weighted MSE emphasizing cruise regime (low accel).
    4. Speed Limit Loss: Penalize v > v_lim.
    5. Non-Negativity Loss: Penalize v < 0.
    6. Acceleration Loss: Supervised loss (dv/dt or vdv/dx).
    """

    def __init__(self, 
                 lambda_peak: float = 0.0,
                 lambda_curv: float = 0.0,
                 use_cruise_weighting: bool = False,
                 alpha_cruise: float = 1.0,
                 accel_proxy: str = 'dvdt',
                 lambda_acc: float = 0.0,
                 lambda_speed_limit: float = 0.0,
                 lambda_nonneg: float = 0.0,
                 lambda_energy: float = 0.0,
                 lambda_power: float = 0.0,
                 lambda_pmp: float = 0.0,
                 limit_alpha: float = 2.0,
                 limit_beta: float = 10.0,
                 limit_gamma: float = 0.85):
        super().__init__()
        self.lambda_peak = lambda_peak
        self.lambda_curv = lambda_curv
        self.lambda_acc = lambda_acc
        self.lambda_speed_limit = lambda_speed_limit
        self.lambda_nonneg = lambda_nonneg
        self.lambda_energy = lambda_energy
        self.lambda_power = lambda_power
        self.lambda_pmp = lambda_pmp
        
        self.limit_alpha = limit_alpha
        self.limit_beta = limit_beta
        self.limit_gamma = limit_gamma
        
        self.use_cruise_weighting = use_cruise_weighting
        self.alpha_cruise = alpha_cruise
        self.accel_proxy = accel_proxy
        
        self.mse = nn.MSELoss(reduction='none')

    def _prepare_sorted(self, x: torch.Tensor, *tensors: torch.Tensor, 
                        DX_MIN_M: float = 1e-3, DX_MAX_M: float = 500.0) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper: Shared Source of Truth for sorting and spatial masking.
        Input:
            x: [B, T] or [B, T, 1]
            tensors: Any number of [B, T, *] tensors aligned with x
        Returns:
            x_sorted: [B, T]
            gathered: List of sorted tensors
            dx: [B, T-1]
            valid_dx: [B, T-1] bool mask (MIN < dx < MAX)
            dx_safe: [B, T-1] clamped dx for division
        """
        if x.dim() == 3: x = x.squeeze(-1)
        B, T = x.shape
        
        # 1. Sort
        x_sorted, idx = torch.sort(x, dim=1, stable=True)
        
        # 2. Gather tensors
        gathered = []
        for t in tensors:
            if t is None:
                gathered.append(None)
                continue
            
            # Match 2D/3D shapes
            if t.dim() == 2:
                gathered.append(t.gather(1, idx))
            elif t.dim() == 3:
                # idx: [B, T] -> [B, T, C]
                C = t.shape[-1]
                idx_3d = idx.unsqueeze(-1).expand(-1, -1, C)
                gathered.append(t.gather(1, idx_3d))
            else:
                gathered.append(t) # Should not happen for standard features

        # 3. Compute spatial gaps
        DX_MIN_M, DX_MAX_M = float(DX_MIN_M), float(DX_MAX_M)
        dx = x_sorted[:, 1:] - x_sorted[:, :-1]
        valid_dx = (dx > DX_MIN_M) & (dx < DX_MAX_M)
        dx_safe = torch.clamp(dx, min=DX_MIN_M)
        
        return x_sorted, gathered, dx, valid_dx, dx_safe

    def compute_curv_loss_v4(self, v_pred: torch.Tensor, x: torch.Tensor, v_gt: Optional[torch.Tensor] = None, 
                             cfg: Dict = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        V4: Curvature loss on non-uniform spatial grid.
        Loss = mean( (d2v / dx_mid)^2 ) over valid triplets.
        """
        cfg = cfg or {}
        DX_MIN_M = float(cfg.get("DX_MIN_M", 1e-3))
        DX_MAX_M = float(cfg.get("DX_MAX_M", 500.0))
        D2V_CLAMP = float(cfg.get("d2v_clamp", 5.0))
        
        # 1. Sort and gather
        xs, gathered, dx, valid_dx, dx_safe = self._prepare_sorted(x, v_pred, v_gt, DX_MIN_M=DX_MIN_M, DX_MAX_M=DX_MAX_M)
        vs = gathered[0]
        vgt = gathered[1]
        
        # 2. First derivative (dv/dx)
        # vs: [B, T, 1] usually. dx_safe: [B, T-1]
        dv = (vs[:, 1:] - vs[:, :-1]) / dx_safe.unsqueeze(-1) # [B, T-1, 1]
        
        # 3. Second derivative (d2v/dx2)
        dx_mid = 0.5 * (dx[:, 1:] + dx[:, :-1]) # [B, T-2]
        dx_mid_safe = torch.clamp(dx_mid, min=DX_MIN_M)
        
        d2v = (dv[:, 1:] - dv[:, :-1]) / dx_mid_safe.unsqueeze(-1) # [B, T-2, 1]
        d2v = torch.clamp(d2v, -D2V_CLAMP, D2V_CLAMP)
        
        # 4. Masking
        valid_d2 = valid_dx[:, 1:] & valid_dx[:, :-1] # [B, T-2]
        
        metrics = {
            "valid_dx_ratio": valid_dx.float().mean().item() if valid_dx.numel() > 0 else 0.0,
            "valid_d2_ratio": valid_d2.float().mean().item() if valid_d2.numel() > 0 else 0.0,
            "dx_min": dx[valid_dx].min().item() if valid_dx.any() else 0.0,
            "dx_median": dx[valid_dx].median().item() if valid_dx.any() else 0.0,
            "dx_max": dx[valid_dx].max().item() if valid_dx.any() else 0.0,
            "zero_dx_ratio": (dx == 0).float().mean().item() if dx.numel() > 0 else 0.0
        }
        
        if not valid_d2.any():
            return torch.tensor(0.0, device=v_pred.device), metrics
            
        # loss = mean differences squared
        loss_raw = torch.mean(d2v[valid_d2]**2)
        
        # Metadata
        metrics["CurvL_raw"] = loss_raw.item()
        
        return self.lambda_curv * loss_raw, metrics

    def compute_speed_limit_loss_v4(self, v_pred: torch.Tensor, v_lim: torch.Tensor, 
                                   v_gt: Optional[torch.Tensor] = None,
                                   mask: Optional[torch.Tensor] = None,
                                   alpha: float = 2.0, beta: float = 10.0, gamma: float = 0.85) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        V4: Overspeed-only loss with optional boundary upweighting.
        """
        # Ensure shapes match [B, T, 1]
        if v_lim.dim() == 2: v_lim = v_lim.unsqueeze(-1)
        if v_gt is not None and v_gt.dim() == 2: v_gt = v_gt.unsqueeze(-1)
        
        overspeed = torch.relu(v_pred - v_lim)
        overspeed2 = overspeed**2
        
        if v_gt is not None:
            # Sigmoid weight: 1.0 + alpha * sigmoid(beta * (v_gt - gamma * v_lim))
            weight = 1.0 + alpha * torch.sigmoid(beta * (v_gt - gamma * v_lim))
            weighted_err = weight * overspeed2
        else:
            weighted_err = overspeed2
            
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            weighted_err = weighted_err * mask.float()
            loss_raw = torch.sum(weighted_err) / mask.float().sum().clamp(min=1.0)
        else:
            loss_raw = torch.mean(weighted_err)
        
        # Metrics
        with torch.no_grad():
            viol_pred = (v_pred > v_lim).float()
            if mask is not None:
                # viol_pred is [B, T, 1], mask is already [B, T, 1] from above
                viol_pred = viol_pred * mask.float()
                
                non_pad_elements = mask.float().sum().clamp(min=1.0)
                lim_frac_pred = viol_pred.sum().item() / non_pad_elements.item()
            else:
                lim_frac_pred = viol_pred.mean().item()
            lim_max_pred = overspeed.max().item()
            
            metrics = {
                "LimFrac_pred": lim_frac_pred,
                "LimMax_pred": lim_max_pred
            }
            
            if v_gt is not None:
                overspeed_gt = torch.relu(v_gt - v_lim)
                metrics["LimFrac_gt"] = (v_gt > v_lim).float().mean().item()
                metrics["LimMax_gt"] = overspeed_gt.max().item()
                
            metrics["LimMean_pred"] = overspeed.mean().item()
                
        return self.lambda_speed_limit * loss_raw, metrics

    def compute_energy_loss_v4(self, v_pred: torch.Tensor, x: torch.Tensor, cfg: Dict = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        V4: Energy proxy loss = mean( (acc^2) * v_mid ) over valid_dx.
        """
        cfg = cfg or {}
        DX_MIN_M = float(cfg.get("DX_MIN_M", 1e-3))
        DX_MAX_M = float(cfg.get("DX_MAX_M", 500.0))
        
        xs, gathered, dx, valid_dx, dx_safe = self._prepare_sorted(x, v_pred, DX_MIN_M=DX_MIN_M, DX_MAX_M=DX_MAX_M)
        vs = gathered[0]
        
        v_mid = 0.5 * (vs[:, 1:] + vs[:, :-1]) # [B, T-1, 1]
        dv_dx = (vs[:, 1:] - vs[:, :-1]) / dx_safe.unsqueeze(-1) # [B, T-1, 1]
        a = v_mid * dv_dx # proxy acceleration
        
        energy = (a**2) * v_mid # [B, T-1, 1]
        
        if not valid_dx.any():
            return torch.tensor(0.0, device=v_pred.device), {"EnergyL_raw": 0.0}
            
        loss_raw = torch.mean(energy[valid_dx])
        metrics = {
            "EnergyL_raw": loss_raw.item(),
            "acc_med": a.abs().median().item(),
            "acc_max": a.abs().max().item()
        }
        return self.lambda_energy * loss_raw, metrics

    def compute_power_loss_v4(self, v_pred: torch.Tensor, x: torch.Tensor, cfg: Dict = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        V4: Power loss with MW scaling and log1p penalty.
        """
        cfg = cfg or {}
        A = float(cfg.get("power_A", 200.0))
        B = float(cfg.get("power_B", 5.0))
        C = float(cfg.get("power_C", 0.1))
        M = float(cfg.get("power_M", 4e5))
        DX_MIN_M = float(cfg.get("DX_MIN_M", 1e-3))
        DX_MAX_M = float(cfg.get("DX_MAX_M", 500.0))
        mode = cfg.get("power_mode", "log1p")
        
        xs, gathered, dx, valid_dx, dx_safe = self._prepare_sorted(x, v_pred, DX_MIN_M=DX_MIN_M, DX_MAX_M=DX_MAX_M)
        vs = gathered[0]
        
        v_mid = 0.5 * (vs[:, 1:] + vs[:, :-1])
        dv_dx = (vs[:, 1:] - vs[:, :-1]) / dx_safe.unsqueeze(-1)
        a = v_mid * dv_dx
        
        f_res = A + B * v_mid + C * (v_mid**2)
        f_trac = M * a + f_res
        p = f_trac * v_mid # Watts-ish
        
        p_mw = p / 1e6 # Scale to MW
        p_pos = torch.relu(p_mw)
        
        if mode == "log1p":
            penalty = torch.log1p(p_pos)
        else:
            penalty = p_pos**2
            
        if not valid_dx.any():
            return torch.tensor(0.0, device=v_pred.device), {"PowerL_raw": 0.0}
            
        loss_raw = torch.mean(penalty[valid_dx])
        
        # Metrics
        with torch.no_grad():
            metrics = {
                "PowerL_raw": loss_raw.item(),
                "P_MW_med": p_mw.median().item(),
                "P_MW_max": p_mw.max().item()
            }
            
        return self.lambda_power * loss_raw, metrics

    def compute_pmp_loss_v4(self, v_pred: torch.Tensor, x: torch.Tensor, 
                           p_mw: Optional[torch.Tensor] = None, 
                           valid_dx: Optional[torch.Tensor] = None,
                           cfg: Dict = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        V4: PMP Residual Loss (Optimality constraint).
        Hamiltonian H = (Power/v) + p * (dv/dx). 
        Optimality: dH/dv = 0.
        """
        cfg = cfg or {}
        if p_mw is None or valid_dx is None:
            # Recompute what's needed
            DX_MIN_M = float(cfg.get("DX_MIN_M", 1e-3))
            DX_MAX_M = float(cfg.get("DX_MAX_M", 500.0))
            A = float(cfg.get("power_A", 200.0))
            B = float(cfg.get("power_B", 5.0))
            C = float(cfg.get("power_C", 0.1))
            M = float(cfg.get("power_M", 4e5))
            
            xs, gathered, dx, valid_dx, dx_safe = self._prepare_sorted(x, v_pred, DX_MIN_M=DX_MIN_M, DX_MAX_M=DX_MAX_M)
            vs = gathered[0]
            v_mid = 0.5 * (vs[:, 1:] + vs[:, :-1])
            dv_dx = (vs[:, 1:] - vs[:, :-1]) / dx_safe.unsqueeze(-1)
            a = v_mid * dv_dx
            f_res = A + B * v_mid + C * (v_mid**2)
            f_trac = M * a + f_res
            p = f_trac * v_mid
            p_mw = p / 1e6
            
            # v_pred for division must match p_mw shape [B, T-1, 1]
            v_div = v_mid
        else:
            v_div = v_pred # Fallback if provided externally
            
        # Simplified residual: P_MW / v_div
        cost_per_m = p_mw / v_div.clamp(min=1.0)
        
        loss_raw = torch.mean(cost_per_m[valid_dx]**2)
        
        return self.lambda_pmp * loss_raw, {"PMPL_raw": loss_raw.item()}

    def compute_peak_loss(self, v_pred: torch.Tensor, v_true: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match vmax."""
        m_exp = mask.unsqueeze(-1)
        v_true_masked = v_true * m_exp
        v_pred_masked = v_pred * m_exp
        v_max_true = torch.max(v_true_masked, dim=1).values
        v_max_pred = torch.max(v_pred_masked, dim=1).values
        loss_raw = torch.mean((v_max_pred - v_max_true)**2)
        return self.lambda_peak * loss_raw, loss_raw

    def compute_nonneg_loss(self, v_pred: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Penalize v < 0."""
        neg_err = torch.relu(-v_pred)
        loss_raw = torch.mean(neg_err**2 * mask.unsqueeze(-1))
        return self.lambda_nonneg * loss_raw, loss_raw

    def compute_curv_loss(self, v_pred: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy Curvature Placeholder (calls V4 with default cfg)."""
        w_loss, metrics = self.compute_curv_loss_v4(v_pred, x, cfg={})
        return w_loss, torch.tensor(metrics.get("CurvL_raw", 0.0), device=v_pred.device)


    def compute_cruise_loss(self, v_pred: torch.Tensor, v_true: torch.Tensor, 
                           accel: Optional[torch.Tensor], 
                           t: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MSE, optionally weighted by cruise regime (low accel).
        """
        err_sq = (v_pred - v_true)**2
        mask_exp = mask.unsqueeze(-1).float()
        
        # Calculate raw MSE for reporting
        raw_mse_sum = torch.sum(err_sq * mask_exp)
        raw_denom = torch.sum(mask_exp)
        raw_loss = raw_mse_sum / raw_denom.clamp(min=1.0)

        if not self.use_cruise_weighting:
            return raw_loss, raw_loss # Weighted is same as raw
            
        # Cruise Weighting
        a_est = None
        
        # 1. Prefer Ground Truth Accel
        if accel is not None:
             a_est = accel.clone()
             if a_est.dim() == 2: a_est = a_est.unsqueeze(-1)
        
        # 2. Fallback Proxy
        if a_est is None:
             if self.accel_proxy == 'dvdt':
                 # Time derivative: (v[i+1]-v[i]) / (t[i+1]-t[i])
                 # Simplified for batch tensor ops:
                 dv = torch.zeros_like(v_true)
                 dv[:, 1:, :] = v_true[:, 1:, :] - v_true[:, :-1, :]
                 
                 dt = torch.zeros_like(v_true) # t is [B, T] -> expand
                 t_exp = t.unsqueeze(-1)
                 dt[:, 1:, :] = t_exp[:, 1:, :] - t_exp[:, :-1, :]
                 dt[dt == 0] = 1.0 # Safety
                 
                 a_est = dv / dt
                 
             elif self.accel_proxy == 'vdvdx':
                 # v * dv/dx
                 dv = torch.zeros_like(v_true)
                 dv[:, 1:, :] = v_true[:, 1:, :] - v_true[:, :-1, :]
                 
                 dx = torch.zeros_like(v_true)
                 x_exp = x.unsqueeze(-1)
                 dx[:, 1:, :] = x_exp[:, 1:, :] - x_exp[:, :-1, :]
                 dx[dx == 0] = 1.0
                 
                 a_est = v_true * (dv / dx)
                 
             else:
                 a_est = torch.zeros_like(v_true) # No weighting possible effectively
        
        # 3. Compute Weights per Trajectory
        weights = torch.ones_like(v_true)
        B = v_true.shape[0]
        
        for b in range(B):
            m_b = mask[b].bool()
            if not m_b.any(): continue
            
            a_b = a_est[b, m_b] # [L, 1]
            
            # Adaptive Threshold: 5% of std(a), min 1e-3
            std_a = torch.std(a_b)
            if torch.isnan(std_a): std_a = torch.tensor(0.0, device=v_true.device)
            eps_a = max(1e-3, 0.05 * std_a.item())
            
            is_cruise = torch.abs(a_est[b]) <= eps_a
            
            # Apply weight
            # Broadcast is_cruise to weights[b] (which is T, 1)
            # is_cruise is (T, 1)
            weights[b] = 1.0 + self.alpha_cruise * is_cruise.float()
            
        # 4. Weighted Loss
        weighted_sq_err = err_sq * weights * mask_exp
        loss_sum = torch.sum(weighted_sq_err)
        denom = torch.sum(weights * mask_exp) # Normalize by sum of weights
        
        if denom < 1e-6:
             return torch.tensor(0.0, device=v_pred.device), raw_loss
             
        weighted_loss = loss_sum / denom
        
        return weighted_loss, raw_loss

    def compute_accel_loss(self, v_pred: torch.Tensor, accel_true: Optional[torch.Tensor], 
                           t: Optional[torch.Tensor], x: Optional[torch.Tensor], mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Supervised Acceleration Loss.
        L_acc = mean( (a_pred - a_true)^2 ) masked.
        
        Proxies:
        - 'dvdt': a_pred = (v_{i+1} - v_i) / (t_{i+1} - t_i) (Requires t)
                  NOTE: Physically inconsistent if v_pred != v_true!
        - 'vdvdx': a_pred = v * dv/dx = v_mid * (v_{i+1} - v_i) / (x_{i+1} - x_i) (Requires x)
                  Physically consistent for spatial trajectories.
        """
        lambda_acc = getattr(self, 'lambda_acc', 0.0) # Safety
        if lambda_acc <= 0:
            return torch.tensor(0.0, device=v_pred.device), torch.tensor(0.0, device=v_pred.device)

        a_pred = None
        
        # Branch based on proxy
        if self.accel_proxy == 'vdvdx':
            if x is None:
                raise ValueError("accel_proxy='vdvdx' requires 'x' argument.")
                
            # 1. Compute dx
            dx = x[:, 1:] - x[:, :-1]
            if dx.dim() == 2: dx = dx.unsqueeze(-1) # Ensure [B, T-1, 1]
            
            # Clamp dx to avoid div by zero (assuming monotonic increasing x)
            # Use a safe epsilon.
            dx = torch.clamp(dx, min=1e-6)
            
            # 2. Compute v_mid
            v_mid = (v_pred[:, 1:] + v_pred[:, :-1]) / 2.0
            
            # 3. Compute dv
            dv = v_pred[:, 1:, :] - v_pred[:, :-1, :]
            
            # 4. a_pred = v * dv/dx
            a_pred = v_mid * (dv / dx)
            
        elif self.accel_proxy == 'dvdt':
            if t is None:
                 raise ValueError("accel_proxy='dvdt' requires 't' argument.")
                 
            # 1. Compute dt
            dt = t[:, 1:] - t[:, :-1]
            dt = torch.clamp(dt, min=1e-6) # Avoid div by zero
            
            # 2. Compute a_pred
            dv = v_pred[:, 1:, :] - v_pred[:, :-1, :]
            a_pred = dv / dt.unsqueeze(-1)
            
        else:
            # Unknown proxy
            return torch.tensor(0.0, device=v_pred.device), torch.tensor(0.0, device=v_pred.device)
        
        # 5. Align a_true
        if accel_true is None:
            # Fallback? No, strict supervised loss requires label.
            # If missing, return 0 (or log warning)
            return torch.tensor(0.0, device=v_pred.device), torch.tensor(0.0, device=v_pred.device)
            
        # accel_true is [B, T, 1] usually. We need to match shape of a_pred [B, T-1, 1].
        # We can take a_true[:, :-1] as "accel at step i" or interpolate.
        # Simple truncation is standard for finite diffs.
        a_tgt = accel_true[:, :-1, :]
        
        # 6. Mask
        # Valid if both i and i+1 are masked
        valid_mask = mask[:, :-1] & mask[:, 1:]
        
        # Also check for numerical stability in a_pred
        # If dx was super small, a_pred might be huge.
        # But we clamped dx.
        
        valid_mask = valid_mask.unsqueeze(-1).float()
        
        # 7. Loss
        err_sq = (a_pred - a_tgt)**2
        weighted_sq = err_sq * valid_mask
        
        loss_sum = torch.sum(weighted_sq)
        denom = torch.sum(valid_mask)
        
        if denom < 1.0:
            return torch.tensor(0.0, device=v_pred.device), torch.tensor(0.0, device=v_pred.device)
            
        raw_loss = loss_sum / denom
        return lambda_acc * raw_loss, raw_loss
