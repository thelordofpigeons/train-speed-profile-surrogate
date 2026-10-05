from typing import Tuple, Optional, Union
import torch
import torch.nn as nn

from .modulated_blocks import ModulatedResidualBlock
from .positional_encoding import SinusoidalPositionalEncoding
from ..utils.bc import compute_xi, apply_hard_bc

class NNCore(nn.Module):
    """
    Pure Function Approximator Core.
    
    Inputs: 
        xi: [B, T, 1] normalized coord
        P: [B, dP] parameters
    
    Output:
        y: [B, T, 1] unconstrained scalar
        
    Architecture:
        PosEnc -> Linear -> [ModulatedResidualBlock] x K -> Linear -> Output
    """
    def __init__(self, param_dim: int, hidden_dim: int = 64, num_blocks: int = 4, num_freqs: int = 16, freq_base: float = 10.0, track_feat_dim: int = 0, num_stations: int = 50, emb_dim: int = 4):
        super().__init__()
        
        self.pos_enc = SinusoidalPositionalEncoding(num_freqs=num_freqs, base=freq_base) 
        self.freq_base = freq_base
        self.track_feat_dim = track_feat_dim
        self.emb_dim = emb_dim
        
        # Station Embedding: Index 0 is UNK/NaN. 1..num_stations are valid.
        # We allow up to num_stations+1 indices.
        self.station_emb = nn.Embedding(num_stations + 1, emb_dim, padding_idx=0)
        
        # Input to MLP: (2 * num_freqs) + track_feat_dim
        input_dim = 2 * num_freqs + track_feat_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Modulated Blocks take P. 
        # P will now be P_feats (continuous) + Station Embs
        # Check P size: param_dim was 8 (legacy). 
        # Now P_eff = (P_cont=6) + (2 * emb_dim).
        # We need to ensure param_dim passed to ModulatedResidualBlock matches this.
        # OR we project P up? ModulatedBlock projects P internally.
        
        self.blocks = nn.ModuleList([
            ModulatedResidualBlock(hidden_dim, param_dim)
            for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, xi: torch.Tensor, P: torch.Tensor, track_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        # P here is the ALREADY FUSED feature vector (Continuous + Embeddings). 
        # Construction happens in SpeedProfileGenerator.
        
        # Positional Encoding
        enc = self.pos_enc(xi) # [B, T, 2N]
        
        if self.track_feat_dim > 0:
            if track_feats is None:
                raise ValueError("Model expects track_feats but none provided.")
            enc = torch.cat([enc, track_feats], dim=-1)
        
        # Project
        h = self.input_proj(enc) # [B, T, H]
        
        # Residual Blocks
        for block in self.blocks:
            h = block(h, P)
            
        # Output
        y = self.output_proj(h) # [B, T, 1]
        return y

class SpeedProfileGenerator(nn.Module):
    """
    System-level wrapper for Speed Profile Generation.
    """
    def __init__(self, param_dim: int, hidden_dim: int = 64, num_blocks: int = 4, 
                 num_freqs: Optional[int] = None, freq_base: Optional[float] = None,
                 track_feat_dim: int = 0, num_stations: int = 100, emb_dim: Union[int, str] = 'auto',
                 extra_features_dim: int = 0):
        super().__init__()
        
        # Enforce explicit Config for Positional Encoding
        if num_freqs is None or freq_base is None:
            raise ValueError("SinusoidalPositionalEncoding requires explicit `num_freqs` and `freq_base`. Check your YAML config.")
            
        # Determine Embedding Dimension
        if str(emb_dim).lower() == 'auto':
            # Auto scaling: min(32, max(4, round(sqrt(num_stations))))
            self.emb_dim = min(32, max(4, int(round(num_stations ** 0.5))))
        else:
            self.emb_dim = int(emb_dim)
            
        # P_eff = (P_cont=6) + (2 * emb_dim) + extra
        self.eff_param_dim = 6 + 2 * self.emb_dim + extra_features_dim
        self.extra_features_dim = extra_features_dim
        
        self.core = NNCore(
            param_dim=self.eff_param_dim, 
            hidden_dim=hidden_dim, 
            num_blocks=num_blocks, 
            num_freqs=num_freqs,
            freq_base=freq_base,
            track_feat_dim=track_feat_dim,
            num_stations=num_stations,
            emb_dim=self.emb_dim
        )
        
    def forward_xi(self, xi: torch.Tensor, P_cont: torch.Tensor, P_raw: torch.Tensor, 
                   station_ids: Optional[torch.Tensor] = None, 
                   track_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            xi: [B, T, 1]
            P_cont: [B, 6] Preprocessed continuous params (ALL dims normalized)
            P_raw: [B, 8] Raw params (for boundary conditions)
            station_ids: [B, 2] LongTensor (StartSt, EndSt)
        """
        if station_ids is None:
            # Fallback (should not happen in correct pipeline)
            device = P_cont.device
            station_ids = torch.zeros((P_cont.size(0), 2), dtype=torch.long, device=device)
            
        # Embeddings
        # station_ids [B, 2] -> emb [B, 2, E] -> flat [B, 2*E]
        embs = self.core.station_emb(station_ids)
        embs_flat = embs.view(embs.size(0), -1)
        
        P_combined = torch.cat([P_cont, embs_flat], dim=1) # [B, 6 + 2E]
        
        # Append Extra Features if present
        if self.extra_features_dim > 0:
            if P_cont.shape[-1] < 6 + self.extra_features_dim:
                # Fallback check handled by caller usually.
                pass
            else:
                 extras = P_cont[:, 6 : 6 + self.extra_features_dim]
                 # We assume P_cont actually passed in includes the extras.
                 # Actually, it's safer if the caller passes exactly `eff_param_dim` total.
                 # Let's cleanly separate: P_cont should be [B, 6] exactly.
                 # If extras are appended, they should be stripped or handled.
                 # In train_h5, we cat [P_feats, log_length, is_seg_batch].
                 # We'll just slice the first 6 for P_cont.
                 pass
                 
        # Re-construct P_combined based on caller assuming P_cont contains extras:
        # P_cont_core = P_cont[:, :6]
        # extras = P_cont[:, 6:]
        # P_combined = cat([P_cont_core, embs_flat, extras])
        
        # Backward compatibility check for parameter schema
        # Legacy schema: P_cont arg is actually P_feats [B, 8 + extra]
        # New continuous schema: P_cont is [B, 6 + extra]
        if P_cont.size(1) == 8 + self.extra_features_dim:
            # Legacy mode
            keep_indices = [0, 1, 2, 3, 6, 7]
            P_cont_core = P_cont[:, keep_indices]
            extras = P_cont[:, 8 : 8 + self.extra_features_dim] if self.extra_features_dim > 0 else torch.empty((P_cont.size(0), 0), device=P_cont.device)
        elif P_cont.size(1) == 6 + self.extra_features_dim:
            # New mode
            P_cont_core = P_cont[:, :6]
            extras = P_cont[:, 6:] if self.extra_features_dim > 0 else torch.empty((P_cont.size(0), 0), device=P_cont.device)
        else:
            raise ValueError(f"Unexpected P_cont size {P_cont.size(1)} (expected {6 + self.extra_features_dim} or {8 + self.extra_features_dim})")
            
        P_combined = torch.cat([P_cont_core, embs_flat, extras], dim=1)
        
        # Track Features Dim Assertion
        if self.core.track_feat_dim > 0:
            if track_feats is None:
                raise ValueError("Model expects track_feats but none provided.")
            if track_feats.shape[-1] != self.core.track_feat_dim:
                raise ValueError(f"Expected track_feats dim {self.core.track_feat_dim}, got {track_feats.shape[-1]}")
                
        # 2. Extract Boundary Values
        v0 = P_raw[:, 2:3]
        vf = P_raw[:, 3:4]
        
        # 3. Call Core
        y = self.core(xi, P_combined, track_feats=track_feats)
        
        # 4. Apply Hard BC
        v = apply_hard_bc(xi, y, v0, vf)
        
        return v
        
    def forward_x(self, x: torch.Tensor, P_feats: torch.Tensor, P_raw: torch.Tensor, 
                  station_ids: Optional[torch.Tensor] = None, 
                  track_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if x.dim() == 2: x = x.unsqueeze(-1)
        elif x.dim() == 1: x = x.view(x.shape[0], 1, 1)
        
        x0 = P_raw[:, 0:1]
        xf = P_raw[:, 1:2]
        
        xi = compute_xi(x, x0, xf)
        
        return self.forward_xi(xi, P_feats, P_raw, station_ids=station_ids, track_feats=track_feats)

    def forward(self, x, P, **kwargs):
        raise NotImplementedError("Legacy forward removed. Use forward_x or forward_xi.")
