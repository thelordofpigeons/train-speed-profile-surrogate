
import os
import json
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Union, Any, Tuple

class SpeedProfileH5Dataset(Dataset):
    """
    PyTorch Dataset for HDF5 Speed Profile Data.
    
    Features:
    - Lazy HDF5 loading (multiprocessing safe).
    - Leakage-free splitting (all segments of a trajectory stay in same split).
    - Mode: 'trajectory' (full) or 'point' (sampled).
    - Track Features: interpolates slope, speed limit, curvature.
    - Acceleration: returns ground truth accel if available.
    """
    
    def __init__(self, 
                 h5_path: str, 
                 split: str = 'train',
                 split_ratio: float = 0.9,
                 seed: int = 42,
                 mode: str = 'point', # 'point' or 'trajectory'
                 points_per_sample: int = 128,
                 force_endpoints_prob: float = 0.3,
                 splits_dir: Optional[str] = None,
                 originals_only: bool = False,
                 use_track_feats: bool = False,
                 **kwargs):
        """
        Args:
            h5_path: Path to dataset.h5
            split: 'train' or 'val'
            split_ratio: Fraction of *original* trajectories in train
            seed: Random seed for splitting and sampling
            mode: 'point' (for training) or 'trajectory' (for analysis)
            points_per_sample: K points to sample in point mode
            force_endpoints_prob: Probability to include t=0 and t=T in sample
            splits_dir: Directory to save/load split JSONs
            originals_only: If True, filter out segments
            use_track_feats: If True, return [K, 3] track features (slope, limit, curv)
            **kwargs: Extra config arguments (use_new_p_schema, use_station_embeddings, use_track_event_features, etc)
        """
        super().__init__()
        self.h5_path = h5_path
        self.split = split
        self.mode = mode
        self.points_per_sample = points_per_sample
        self.force_endpoints_prob = force_endpoints_prob
        self.seed = seed
        self.originals_only = originals_only
        self.use_track_feats = use_track_feats
        
        # Multiprocessing: Do NOT open file here
        self.h5_file = None
        
        # Track Features Cache (Loaded once, small enough for memory)
        self.track_data = {}
        if self.use_track_feats:
            self._load_track_data()
        
        # Setup Splits
        if splits_dir is None:
            self.splits_dir = os.path.join(os.path.dirname(h5_path), 'splits')
        else:
            self.splits_dir = splits_dir
            
        os.makedirs(self.splits_dir, exist_ok=True)
        self.group_names = self._setup_splits(split, split_ratio, seed)
        
        # Filter if requested
        if self.originals_only:
            pre_filter_len = len(self.group_names)
            self.group_names = [n for n in self.group_names if "_seg_" not in n]
            print(f"H5Dataset({split}): Filtered {pre_filter_len} -> {len(self.group_names)} samples (originals_only=True)")
            
        # V3 Tail Fixes: Station ID mapping & Trajectory Stats Caching
        self.use_new_p_schema = kwargs.get('use_new_p_schema', False)
        self.use_station_embeddings = kwargs.get('use_station_embeddings', False)
        self.use_track_event_features = kwargs.get('use_track_event_features', False)
        
        # 1. Map Station IDs & Cache Traj Stats
        self.unique_station_ids = []
        self.id_to_idx = {}
        self.num_stations = 1 # Index 0 is reserved for UNK
        self.UNK_STATION_IDX = 0
        self.unknown_station_count = 0
        
        self.L_cache = np.zeros(len(self.group_names), dtype=np.float32)
        self.vsegmax_cache = np.zeros(len(self.group_names), dtype=np.float32)
        
        if self.use_new_p_schema or self.use_station_embeddings:
            self._init_mappings_and_stats()
            
        self.is_trajectory_level = (self.mode == 'trajectory')
        
        print(f"H5Dataset({split}): Loaded {len(self.group_names)} samples. Mode={mode}, TrackFeats={use_track_feats}")

    def get_traj_stats(self, idx: int) -> Tuple[float, float]:
        """Returns precomputed (L, v_seg_max) for stratified sampling."""
        return self.L_cache[idx], self.vsegmax_cache[idx]

    def _init_mappings_and_stats(self):
        print("H5Dataset: Precomputing trajectory stats and mapping station IDs...")
        all_raw_ids = []
        
        with h5py.File(self.h5_path, 'r', swmr=True) as f:
            for i, group_name in enumerate(tqdm(self.group_names, desc="Init Dataset Scans")):
                g = f[group_name]
                
                # Fetch P params
                P_arr = g['P'][:]
                x0 = P_arr[0]
                xf = P_arr[1]
                L = xf - x0
                assert L > 0, f"Trajectory {group_name} has non-positive length L={L}"
                self.L_cache[i] = L
                
                # Precompute v_seg_max
                # Needs x array and track data if interpolating
                if self.use_track_feats and 'spdLimit' in self.track_data:
                    # Query max speed limit over the trajectory space.
                    # We interpolate at train time, but for cache we can check physical limits
                    x_arr = g['x'][:]
                    v_max = np.max(self._interpolate_track_features(x_arr)[:, 1])
                else:
                    v_max = max(P_arr[2], P_arr[3], 15.0) # Fallback heuristic if no track feats
                    
                self.vsegmax_cache[i] = v_max
                
                # Collect station IDs
                st_ids = np.nan_to_num(P_arr[[4, 5]], nan=0.0).astype(np.int64)
                all_raw_ids.extend(st_ids.tolist())
                
        # deterministic mapping
        unique_ids = sorted(list(set(all_raw_ids)))
        self.unique_station_ids = unique_ids
        
        # 1-indexed (0 is UNK)
        self.id_to_idx = {raw_id: i+1 for i, raw_id in enumerate(unique_ids)}
        self.num_stations = len(unique_ids) + 1
        
        # Persist mapping to artifacts
        os.makedirs("artifacts", exist_ok=True)
        out_path = os.path.join("artifacts", f"station_id_map_{self.split}.json")
        with open(out_path, "w") as map_f:
            json.dump({"id_to_idx": self.id_to_idx, "UNK_IDX": 0}, map_f, indent=2)
            
        print(f"H5Dataset: Mapped {self.num_stations - 1} unique stations. (Total embedded {self.num_stations})")

    def _load_track_data(self):
        """Load global railTrack arrays into memory."""
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if 'railTrack' in f:
                    rt = f['railTrack']
                    for key in ['slope', 'spdLimit', 'curvature']:
                        if key in rt:
                            self.track_data[key] = rt[key][:]
                        else:
                            print(f"Warning: railTrack/{key} missing in H5.")
                else:
                    print("Warning: /railTrack group missing in H5.")
        except Exception as e:
            print(f"Failed to load track data: {e}")

    def _interpolate_track_features(self, x_phys: np.ndarray, x0: float = 0.0, L: float = 1.0, xi_phys: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Interpolate slope, limit, curvature at physical positions x_phys.
        If self.use_track_event_features is True, compute advanced features (needs x0, L, xi_phys).
        Returns: [K, F] array.
        """
        # Lookahead offsets (scale-invariant)
        # For base features we sample at: xi' = clamp(xi + dxi)
        # Base lookahead offsets: 7 points -> 21 dims
        dxi_base = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.7])
        
        # Determine number of base and total features
        base_dim = 3 * len(dxi_base) # 21
        
        event_dim = 0
        use_events = getattr(self, 'use_track_event_features', False)
        if use_events:
            event_dim = 5
            
        total_dim = base_dim + event_dim
        
        feats = np.zeros((len(x_phys), total_dim), dtype=np.float32)

        # If we need lookahead, we need xi. If xi not provided (e.g. cache init), fallback to x_phys directly for 1 point.
        if xi_phys is None:
            # Fallback path for simple queries (e.g. init cache max vlim)
             # Just query 0.0 lookahead
             feats_fallback = np.zeros((len(x_phys), 3), dtype=np.float32)
             if 'slope' in self.track_data:
                 dat = self.track_data['slope']
                 feats_fallback[:, 0] = np.interp(x_phys, dat[:, 0], dat[:, 1])
             if 'spdLimit' in self.track_data:
                 dat = self.track_data['spdLimit']
                 feats_fallback[:, 1] = np.interp(x_phys, dat[:, 0], dat[:, 1])
             if 'curvature' in self.track_data:
                 dat = self.track_data['curvature']
                 feats_fallback[:, 2] = np.interp(x_phys, dat[:, 0], dat[:, 1])
             
             # Pad to total_dim if needed (though usually we just want vlim here)
             feats[:, :3] = feats_fallback
             return feats

        # Main Forward Path with Lookahead
        # xi_phys is [K]
        
        for k in range(len(xi_phys)):
            xi = xi_phys[k]
            
            # --- BASE LOOKAHEAD (21 dims) ---
            for i, dxi in enumerate(dxi_base):
                xi_prime = np.clip(xi + dxi, 0.0, 1.0)
                x_query = x0 + xi_prime * L
                
                if 'slope' in self.track_data:
                    dat = self.track_data['slope']
                    feats[k, i*3 + 0] = np.interp(x_query, dat[:, 0], dat[:, 1])
                if 'spdLimit' in self.track_data:
                    dat = self.track_data['spdLimit']
                    feats[k, i*3 + 1] = np.interp(x_query, dat[:, 0], dat[:, 1])
                if 'curvature' in self.track_data:
                    dat = self.track_data['curvature']
                    feats[k, i*3 + 2] = np.interp(x_query, dat[:, 0], dat[:, 1])
                    
            # --- EVENT FEATURES (5 dims) ---
            if use_events:
                event_horizon = getattr(self, 'event_horizon', 0.4)
                steps = getattr(self, 'event_grid_steps', 33)
                
                # Grid
                deltas = np.linspace(0.0, event_horizon, steps)
                xi_grid = np.clip(xi + deltas, 0.0, 1.0)
                x_grid = x0 + xi_grid * L
                
                # Query on grid
                sl_grid = np.interp(x_grid, self.track_data['spdLimit'][:, 0], self.track_data['spdLimit'][:, 1])
                cv_grid = np.interp(x_grid, self.track_data['curvature'][:, 0], self.track_data['curvature'][:, 1]) if 'curvature' in self.track_data else np.zeros_like(x_grid)
                
                vlim0 = sl_grid[0]
                
                # min_future_speed_limit
                min_vlim = np.min(sl_grid)
                
                # max_future_curvature
                max_curv = np.max(cv_grid)
                
                # distance to drop
                eps = max(0.5, 0.02 * vlim0)
                drop_idx = np.where(sl_grid < vlim0 - eps)[0]
                
                if len(drop_idx) > 0:
                    first_idx = drop_idx[0]
                    dist_drop = deltas[first_idx]
                    has_drop = 1.0
                    drop_mag = vlim0 - sl_grid[first_idx]
                else:
                    dist_drop = min(event_horizon, 1.0 - xi)  # NOT 1.0
                    has_drop = 0.0
                    drop_mag = 0.0
                    
                idx_base = base_dim
                feats[k, idx_base + 0] = min_vlim
                feats[k, idx_base + 1] = max_curv
                feats[k, idx_base + 2] = dist_drop
                feats[k, idx_base + 3] = has_drop
                feats[k, idx_base + 4] = drop_mag

        return feats

    def _setup_splits(self, split: str, ratio: float, seed: int) -> List[str]:
        """Load or create splits ensuring no leakage."""
        train_file = os.path.join(self.splits_dir, 'train_split.json')
        val_file = os.path.join(self.splits_dir, 'val_split.json')
        
        if os.path.exists(train_file) and os.path.exists(val_file):
            if split == 'train':
                with open(train_file, 'r') as f: return json.load(f)
            else:
                with open(val_file, 'r') as f: return json.load(f)
        
        print(f"Creating new splits (ratio={ratio}, seed={seed})...")
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if 'index' in f:
                    index = f['index'][:]
                    all_names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in index]
                else:
                    # Fallback if index missing (e.g. iterate groups)
                    all_names = [k for k in f.keys() if k.startswith('traj_')]
        except Exception as e:
            print(f"Error reading index: {e}")
            return []
            
        base_groups = {}
        for name in all_names:
            base_id = name.split('_seg_')[0]
            if base_id not in base_groups: base_groups[base_id] = []
            base_groups[base_id].append(name)
            
        base_ids = sorted(list(base_groups.keys()))
        rng = np.random.RandomState(seed)
        rng.shuffle(base_ids)
        
        n_train = int(len(base_ids) * ratio)
        train_ids = set(base_ids[:n_train])
        val_ids = set(base_ids[n_train:])
        
        train_names = []
        for bid in train_ids: train_names.extend(base_groups[bid])
            
        val_names = []
        for bid in val_ids: val_names.extend(base_groups[bid])
            
        with open(train_file, 'w') as f: json.dump(train_names, f)
        with open(val_file, 'w') as f: json.dump(val_names, f)
        
        if split == 'train': return train_names
        return val_names

    def __getstate__(self):
        state = self.__dict__.copy()
        state['h5_file'] = None
        return state

    def _get_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
        return self.h5_file

    def __len__(self):
        return len(self.group_names)
        
    def __getitem__(self, idx):
        group_name = self.group_names[idx]
        f = self._get_file()
        g = f[group_name]
        
        # Load Raw Data
        t_arr = g['t'][:]
        x_arr = g['x'][:]
        v_arr = g['v'][:]
        xi_arr = g['xi'][:]
        P_arr = g['P'][:] # (8,)
        
        accel_arr = None
        if 'accel' in g:
            accel_arr = g['accel'][:]
        
        # Robust is_segment determination
        is_segment = 1.0 if "_seg_" in group_name else 0.0

        # Map Station IDs
        st_ids = P_arr[4:6].astype(np.int64)
        mapped_st_ids = np.zeros(2, dtype=np.int64)
        if hasattr(self, 'id_to_idx'):
            for i, raw_id in enumerate(st_ids):
                if raw_id in self.id_to_idx:
                    mapped_st_ids[i] = self.id_to_idx[raw_id]
                else:
                    mapped_st_ids[i] = self.UNK_STATION_IDX
                    self.unknown_station_count += 1
                    
            assert np.all((mapped_st_ids >= 0) & (mapped_st_ids < self.num_stations)), "Mapped ID out of bounds"
        else:
            mapped_st_ids = st_ids # Fallback

        # Continuous Vector (P_cont)
        if self.use_new_p_schema:
            x0 = P_arr[0]
            xf = P_arr[1]
            L = xf - x0
            v0 = P_arr[2]
            vf = P_arr[3]
            dur = P_arr[6]
            M = P_arr[7]
            v_seg_max = self.vsegmax_cache[idx] if hasattr(self, 'vsegmax_cache') else 0.0
            
            p_cont = np.array([L, v0, vf, dur, M, v_seg_max], dtype=np.float32)
            
        if self.mode == 'trajectory':
            # Track Features (Full)
            track_feats = None
            if self.use_track_feats:
                x0_b = P_arr[0]
                L_b = max(P_arr[1] - P_arr[0], 1e-6)
                track_feats = self._interpolate_track_features(x_arr, x0=x0_b, L=L_b, xi_phys=xi_arr)

            item = {
                "group": group_name,
                "t": torch.as_tensor(t_arr, dtype=torch.float32),
                "x": torch.as_tensor(x_arr, dtype=torch.float32),
                "v": torch.as_tensor(v_arr, dtype=torch.float32),
                "xi": torch.as_tensor(xi_arr, dtype=torch.float32),
                "P": torch.as_tensor(P_arr, dtype=torch.float32),
                "station_ids": torch.as_tensor(mapped_st_ids, dtype=torch.long),
                "is_segment": torch.tensor([is_segment], dtype=torch.float32)
            }
            if self.use_new_p_schema:
                item["p_cont"] = torch.as_tensor(p_cont, dtype=torch.float32)
                item["st_start"] = torch.tensor(mapped_st_ids[0], dtype=torch.long)
                item["st_end"] = torch.tensor(mapped_st_ids[1], dtype=torch.long)
                
            if accel_arr is not None:
                item["accel"] = torch.as_tensor(accel_arr, dtype=torch.float32)
            
            if track_feats is not None:
                item["track_feats"] = torch.as_tensor(track_feats, dtype=torch.float32)
                
            return item
            
        elif self.mode == 'point':
            N = len(t_arr)
            K = self.points_per_sample
            
            # Sampling
            indices = np.random.randint(0, N, size=K)
            # Force Endpoints logic
            if np.random.rand() < self.force_endpoints_prob:
                indices[0] = 0
                indices[1] = N - 1
            # Sort for valid trajectory snippet
            indices.sort()
            
            x_sample = x_arr[indices]
            v_sample = v_arr[indices]
            xi_sample = xi_arr[indices]
            
            item = {
                "x": torch.tensor(x_sample, dtype=torch.float32).unsqueeze(-1),
                "v": torch.tensor(v_sample, dtype=torch.float32).unsqueeze(-1),
                "xi": torch.tensor(xi_sample, dtype=torch.float32).unsqueeze(-1),
                "P": torch.tensor(P_arr, dtype=torch.float32),
                "station_ids": torch.as_tensor(mapped_st_ids, dtype=torch.long),
                "group": group_name,
                "is_segment": torch.tensor([is_segment], dtype=torch.float32)
            }
            
            if self.use_new_p_schema:
                item["p_cont"] = torch.as_tensor(p_cont, dtype=torch.float32)
                item["st_start"] = torch.tensor(mapped_st_ids[0], dtype=torch.long)
                item["st_end"] = torch.tensor(mapped_st_ids[1], dtype=torch.long)
            
            if accel_arr is not None:
                item["accel"] = torch.tensor(accel_arr[indices], dtype=torch.float32).unsqueeze(-1)

            if self.use_track_feats:
                x0_b = P_arr[0]
                L_b = max(P_arr[1] - P_arr[0], 1e-6)
                track_feats = self._interpolate_track_features(x_sample, x0=x0_b, L=L_b, xi_phys=xi_sample)
                item["track_feats"] = torch.tensor(track_feats, dtype=torch.float32)
                
            return item
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

            

