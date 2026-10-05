import torch
import yaml
import json
import os
import numpy as np
from src.speed_profile_generator.models.speed_profile_generator import SpeedProfileGenerator
from src.speed_profile_generator.utils.preprocessor import PPreprocessor

class Planner:
    """
    Interfaces with the Neural Surrogate Model to generate speed profiles.
    Measurements in SI units (m, m/s).
    """
    def __init__(self, run_dir, ckpt_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.run_dir = run_dir
        self.ckpt_path = ckpt_path
        self.model = None
        self.preprocessor = None
        self.config = None
        self.track_stats = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        # 1. Config
        # If run_dir is None, we might infer it from ckpt_path?
        # But for now assume run_dir is always provided for config context.
        config_path = os.path.join(self.run_dir, "config.yaml")
        if not os.path.exists(config_path):
             # Fallback or error
             raise FileNotFoundError(f"Config not found at {config_path}")
             
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 2. Stats for P normalization
        stats_path = os.path.join(self.run_dir, "norm_stats.json")
        self.preprocessor = PPreprocessor()
        self.preprocessor.load(stats_path)
        
        # 3. Track Stats (if available)
        ts_path = os.path.join(self.run_dir, "track_stats.json")
        if os.path.exists(ts_path):
            with open(ts_path, 'r') as f:
                self.track_stats = json.load(f)
                
        # 4. Model
        # Use provided ckpt_path or find in run_dir
        if self.ckpt_path:
            ckpt_path = self.ckpt_path
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        else:
            ckpt_path = os.path.join(self.run_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            import glob
            pts = glob.glob(os.path.join(self.run_dir, "*.pt"))
            if pts: ckpt_path = pts[0]
            else: raise FileNotFoundError("No checkpoint found.")
            
        m_conf = self.config['model']
        self.model = SpeedProfileGenerator(
            param_dim=m_conf['param_dim'],
            hidden_dim=m_conf['hidden_dim'],
            num_blocks=m_conf['num_blocks'],
            track_feat_dim=m_conf.get('track_feat_dim', 0),
            emb_dim=m_conf.get('emb_dim', 4),
            num_freqs=m_conf.get('num_freqs', 16),
            freq_base=m_conf.get('freq_base', 10.0),
            extra_features_dim=m_conf.get('extra_features_dim', 0)
        )
        
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(self.device)
        print(f"Planner loaded from {self.run_dir} on {self.device}")

    def plan(self, x0, v0, xf, vf, track_feats=None, P_ref=None, dur_policy='keep', delay=0.0):
        """
        Generate a speed profile v(x) from x0 to xf.
        P_ref: Optional [8] or [1, 8] tensor/numpy array containing physics context.
        Returns: (x_grid, v_grid)
        """
        # Create grid
        # We model continuous function but evaluate on grid for simulation/vis
        # Use fixed resolution, e.g. 1m or 100 points?
        # Model is Resolution Independent.
        # Let's pick 128 points for query
        K = 128
        x_grid = torch.linspace(x0, xf, K, device=self.device).reshape(1, K, 1) # [1, K, 1]
        
        # P Construction
        if P_ref is None:
            raise ValueError("P_ref is required for planning to ensure correct physics context (Mass, Duration, etc.).")
            
        # Handle P_ref input types
        if isinstance(P_ref, np.ndarray):
            P_raw = torch.from_numpy(P_ref).float().to(self.device)
        elif isinstance(P_ref, torch.Tensor):
            P_raw = P_ref.clone().float().to(self.device)
        else:
            P_raw = torch.tensor(P_ref, device=self.device, dtype=torch.float32)
            
        # Ensure shape [1, 8]
        if P_raw.dim() == 1:
            P_raw = P_raw.unsqueeze(0)
            
        # Overwrite State Variables ONLY
        # Indices: 0:x0, 1:xf, 2:v0, 3:vf
        P_raw[0, 0] = float(x0)
        P_raw[0, 1] = float(xf)
        P_raw[0, 2] = float(v0)
        P_raw[0, 3] = float(vf)
        
        # --- FIX: Update Duration for Replanning ---
        # If x0 has moved significantly, the original 'dur' (total time) is no longer valid for the subtrajectory.
        # We need to estimate a reasonable duration for the remaining distance (xf - x0).
        # We can use a heuristic based on the original average speed, or a fixed assumption.
        
        # --- FIX: Update Duration for Replanning with Policy ---
        # 1. Get original P context
        if isinstance(P_ref, torch.Tensor):
            x0_ref = P_ref[0, 0].item() if P_ref.dim() == 2 else P_ref[0].item()
            xf_ref = P_ref[0, 1].item() if P_ref.dim() == 2 else P_ref[1].item()
            dur_ref = P_ref[0, 6].item() if P_ref.dim() == 2 else P_ref[6].item()
        else:
            P_np = np.array(P_ref)
            if P_np.ndim == 2:
                x0_ref, xf_ref, dur_ref = float(P_np[0, 0]), float(P_np[0, 1]), float(P_np[0, 6])
            else:
                x0_ref, xf_ref, dur_ref = float(P_np[0]), float(P_np[1]), float(P_np[6])

        L_ref = xf_ref - x0_ref
        L_rem = xf - x0
        
        # Apply Policy
        # dur_policy: 'keep' (default) or 'recover'
        
        if dur_policy == 'keep':
            # ENERGY OPTIMAL CONTINUATION
            # Use original duration (unscaled). This means we accept delay.
            dur_final = dur_ref
            print(f"Replanning [Keep]: Energy-optimal continuation. Fixed Duration {dur_final:.1f}s")
            
        elif dur_policy == 'recover':
            # Schedule Recovery
            # Reduce duration to catch up.
            # Base it on proportional time? Or remaining time?
            # Let's assume we want to maintain the original average speed for the REMAINING distance, minus delay.
             
            if L_ref > 1.0 and dur_ref > 0.1:
                ratio = L_rem / L_ref
                dur_nominal = dur_ref * ratio
            else:
                dur_nominal = dur_ref
                
            dur_final = max(1.0, dur_nominal - delay)
            print(f"Replanning [Recover]: Nominal {dur_nominal:.1f}s - Delay {delay:.1f}s -> {dur_final:.1f}s")
        else:
            # Fallback
            dur_final = dur_ref
            
        # Update P
        P_raw[0, 6] = float(dur_final)
        # --- End FIX ---
        
        # --- End FIX ---
        
        # Validation
        M = P_raw[0, 7].item()
        dur = P_raw[0, 6].item()
        
        if M < 1000:
            print(f"WARNING: Mass {M} seems too low! Expected ~3e5.")
            # assert M > 1000, f"Mass {M} is unrealistically low." # Soft warn for now or hard fail? User said assert.
            # Let's hard fail to be safe as requested "assert P[7] > 1000"
            raise ValueError(f"Mass {M} is unrealistically low (Expected > 1000). Check P_ref source.")
            
        if dur <= 0:
            raise ValueError(f"Duration {dur} must be positive.")
            
        if xf <= x0:
            print(f"Warning: xf ({xf}) <= x0 ({x0}). Profile might be degenerate.")

        # Debug Logging
        print(f"Planning P: x0={x0:.1f}, xf={xf:.1f}, v0={v0:.1f}, vf={vf:.1f}, "
              f"St={int(P_raw[0,4].item())}->{int(P_raw[0,5].item())}, "
              f"dur={dur:.1f}, M={M:.0f}")
        
        # Preprocess
        P_feats = self.preprocessor.transform(P_raw)
        
        # Extras: log_length, is_seg
        length = max(xf - x0, 1e-6)
        log_len = torch.tensor([[np.log(length)]], device=self.device, dtype=torch.float32)
        is_seg = torch.zeros((1, 1), device=self.device, dtype=torch.float32) # Assume full traj context for robustness? Or segment?
        
        # If we are replanning mid-way, it conceptually IS a segment or a new problem with boundary conditions.
        # The model was trained on segments with is_segment=1.
        # But also full trajectories.
        # Let's try is_segment=1 if it's a "replan" (short horizon) or just 0 if we treat it as a new full problem.
        # Actually, if we just change x0/v0, it IS a new problem defined by P.
        # Let's keep is_segment=0 for now unless results are bad. The model generalizes P.
        
        # WAIT: The model expects specific extra dimensions.
        # If config has extra_features_dim=2, we need log_len and is_seg.
        if self.config['model'].get('extra_features_dim', 0) == 2:
            P_feats = torch.cat([P_feats, log_len, is_seg], dim=-1)
            
        # Track Features
        tf_norm = None
        if track_feats is not None:
            # Convert to tensor if needed
            if isinstance(track_feats, np.ndarray):
                track_feats = torch.from_numpy(track_feats).float().to(self.device).unsqueeze(0) # [1, K, 3]

            # track_feats should be [1, K, 3]
            if track_feats.dim() == 2:
                track_feats = track_feats.unsqueeze(0)

            # Normalize
            if self.track_stats:
                mean = torch.tensor(self.track_stats['mean'], device=self.device).view(1, 1, -1)
                std = torch.tensor(self.track_stats['std'], device=self.device).view(1, 1, -1)

                std[std < 1e-6] = 1.0
                tf_norm = (track_feats - mean) / std
            else:
                tf_norm = track_feats
        
        # Inference
        with torch.no_grad():
            # forward_x expects P_feats, P_raw, etc.
            # station_ids=None defaults to 0
            v_pred = self.model.forward_x(x_grid, P_feats, P_raw, track_feats=tf_norm)
            
        return x_grid.squeeze().cpu().numpy(), v_pred.squeeze().cpu().numpy()
