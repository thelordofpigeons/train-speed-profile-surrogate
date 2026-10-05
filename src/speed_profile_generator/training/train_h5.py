import os
import sys
import yaml
import argparse
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from speed_profile_generator.data.h5_dataset import SpeedProfileH5Dataset
from speed_profile_generator.models.speed_profile_generator import SpeedProfileGenerator
from speed_profile_generator.utils.preprocessor import PPreprocessor
from speed_profile_generator.training.losses_v3 import SpeedProfileLossV3

def pad_trajectory_batch(batch):
    """
    Collate function for variable length trajectories.
    Batch is list of dicts: {'t', 'x', 'v', 'xi', 'P', 'accel' (opt), 'track_feats' (opt), 'group'}.
    """
    max_len = 0
    for item in batch:
        max_len = max(max_len, item['t'].shape[0])
        
    B = len(batch)
    
    # Init 
    mask = torch.zeros((B, max_len), dtype=torch.bool)
    
    t_batch = torch.zeros((B, max_len), dtype=torch.float32)
    x_batch = torch.zeros((B, max_len), dtype=torch.float32)
    v_batch = torch.zeros((B, max_len, 1), dtype=torch.float32)
    xi_batch = torch.zeros((B, max_len, 1), dtype=torch.float32)
    
    has_accel = any('accel' in item for item in batch)
    accel_batch = None
    if has_accel:
        accel_batch = torch.zeros((B, max_len, 1), dtype=torch.float32)
        
    has_tf = any('track_feats' in item for item in batch)
    tf_batch = None
    if has_tf:
        # Assuming track feats dim is 3
        # item['track_feats'] should be [T, 3] or [T, F]
        # Inspect first one
        tf_dim = 3
        for item in batch:
            if 'track_feats' in item:
                tf_dim = item['track_feats'].shape[-1]
                break
        tf_batch = torch.zeros((B, max_len, tf_dim), dtype=torch.float32)
        
    has_stations = any('station_ids' in item for item in batch)
    st_batch = torch.zeros((B, 2), dtype=torch.long) # Global per traj, not seq

        
    P_list = []
    groups = []
    is_seg_list = [] # For generalization features
    
    for i, item in enumerate(batch):
        L = item['t'].shape[0]
        mask[i, :L] = True
        
        t_batch[i, :L] = item['t']
        x_batch[i, :L] = item['x']
        
        v_i = item['v']
        if v_i.dim() == 1: v_i = v_i.unsqueeze(-1)
        v_batch[i, :L] = v_i
        
        xi_i = item['xi']
        if xi_i.dim() == 1: xi_i = xi_i.unsqueeze(-1)
        xi_batch[i, :L] = xi_i
        
        if has_accel and 'accel' in item:
            a_i = item['accel']
            if a_i.dim() == 1: a_i = a_i.unsqueeze(-1)
            accel_batch[i, :L] = a_i
            
        if has_tf and 'track_feats' in item:
            tf_i = item['track_feats']
            tf_batch[i, :L, :] = tf_i
            
        if has_stations and 'station_ids' in item:
            st_batch[i] = item['station_ids']
            
        P_list.append(item['P'])
        groups.append(item['group'])
        
        if 'is_segment' in item:
            is_seg_list.append(item['is_segment'])
        else:
             is_seg_list.append(torch.tensor([0.0]))
        
    P_batch = torch.stack(P_list) # [B, 8]
    
    out = {
        "t": t_batch,
        "x": x_batch,
        "v": v_batch,
        "xi": xi_batch,
        "P": P_batch,
        "is_segment": torch.stack(is_seg_list) if is_seg_list else torch.zeros(B, 1)
    }
    
    if accel_batch is not None:
        out['accel'] = accel_batch
        
    if tf_batch is not None:
        out['track_feats'] = tf_batch

    if has_stations:
        out['station_ids'] = st_batch
        
    out['mask'] = mask
    out['group'] = groups
    
    return out

def compute_track_stats(dataloader, device='cpu'):
    """
    Compute mean/std of track features [slope, limit, curv] on training set.
    """
    print("Computing track feature statistics on training set...")
    count = 0
    mean_sum = None
    sq_sum = None
    
    min_val = None
    max_val = None
    
    for i, batch in enumerate(tqdm(dataloader, desc="TrackStats")):
        if i >= 10: break # Only use 10 batches (approx 2560 points) for stats to speed up init
        # Point mode: batch is dict. 'track_feats' is [B, 3] or [B, K, 3] depending on mode.
        # H5Dataset point mode returns 'track_feats': [3] per sample?
        # Let's check collator or dataset.
        # H5Dataset point mode returns item['track_feats'] which is array(3).
        # Default loader collates to [B, 3].
        
        if 'track_feats' not in batch:
            continue
            
        tf = batch['track_feats'].to(device) # [B, F] or [B, T, F]
        if tf.dim() == 2:
            pass
        elif tf.dim() == 3:
            tf = tf.reshape(-1, tf.shape[-1])
            
        mask = torch.isfinite(tf).all(dim=1)
        if not mask.any(): continue
        
        tf = tf[mask]
        feat_dim = tf.shape[1]
        
        if mean_sum is None:
            mean_sum = torch.zeros(feat_dim, device=device)
            sq_sum = torch.zeros(feat_dim, device=device)
            min_val = torch.full((feat_dim,), float('inf'), device=device)
            max_val = torch.full((feat_dim,), float('-inf'), device=device)
            
        count += tf.shape[0]
        mean_sum += tf.sum(dim=0)
        sq_sum += (tf**2).sum(dim=0)
        
        min_val = torch.minimum(min_val, tf.min(dim=0).values)
        max_val = torch.maximum(max_val, tf.max(dim=0).values)
        
    if count == 0 or mean_sum is None:
        print("Warning: No track features found statistics.")
        return {'mean': [0.0]*21, 'std': [1.0]*21} # fallback dimension 21
        
    mean = mean_sum / count
    # std = sqrt(E[x^2] - E[x]^2)
    var = (sq_sum / count) - (mean ** 2)
    var = torch.clamp(var, min=1e-6)
    std = torch.sqrt(var)
    
    stats = {
        'mean': mean.cpu().tolist(),
        'std': std.cpu().tolist(),
        'min': min_val.cpu().tolist(),
        'max': max_val.cpu().tolist(),
        'count': count,
        'timestamp': datetime.now().isoformat(),
        'h5_path': dataloader.dataset.h5_path if hasattr(dataloader.dataset, 'h5_path') else 'unknown'
    }
    
    print(f"Track Stats Computed (N={count}):")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std:  {stats['std']}")
    print(f"  Min:  {stats['min']}")
    print(f"  Max:  {stats['max']}")
    return stats

def normalize_track_feats(track_feats, stats, device):
    """
    (tf - mean) / (std + eps)
    """
    if track_feats is None: return None
    
    mu = torch.tensor(stats['mean'], device=device, dtype=track_feats.dtype)
    sigma = torch.tensor(stats['std'], device=device, dtype=track_feats.dtype)
    
    # Broadcast
    # if tf is [B, 3] -> mu is [3] ok
    # if tf is [B, T, 3] -> mu is [3] ok broadcast last dim
    
    return (track_feats - mu) / (sigma + 1e-6)

def setup_logging(run_dir):
    log_path = os.path.join(run_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def save_plots(model, dataset, preprocessor, run_dir, device, epoch, num_plots=20, track_stats=None):
    model.eval()
    plot_dir = os.path.join(run_dir, 'plots', f'epoch_{epoch:03d}')
    os.makedirs(plot_dir, exist_ok=True)
    
    original_indices = [i for i, name in enumerate(dataset.group_names) if "_seg_" not in name]
    if not original_indices: return
        
    indices = original_indices[:num_plots]
    rmses = []
    
    with torch.no_grad():
        for idx in indices:
            item = dataset[idx]
            group_name = item['group']
            
            x = item['x'].to(device)
            v_true = item['v'].to(device)
            P_raw = item['P'].to(device).unsqueeze(0)
            
            # Augment P for plotting
            x0 = P_raw[:, 0]
            xf = P_raw[:, 1]
            length = torch.clamp(xf - x0, min=1e-6)
            log_length = torch.log(length).unsqueeze(-1) # [1, 1]
            is_seg_batch = torch.tensor([1.0 if "_seg_" in group_name else 0.0], device=device).unsqueeze(-1) # [1, 1]
            
            # Extract Continuous Physical Parameters for Preprocessor
            P_cont = P_raw[:, :6]
            P_feats = preprocessor.transform_cont(P_cont)
            P_feats = torch.cat([P_feats, log_length, is_seg_batch], dim=-1) # [1, 10]
            
            track_feats = None
            if 'track_feats' in item:
                track_feats = item['track_feats'].to(device).unsqueeze(0) # [1, T, 3]
                if track_stats is not None:
                     track_feats = normalize_track_feats(track_feats, track_stats, device)
            
            if x.dim() == 1: x_in = x.unsqueeze(0)
            else: x_in = x
            
            st_ids = None
            if 'station_ids' in item:
                st_ids = item['station_ids'].to(device).unsqueeze(0)
                
            v_pred = model.forward_x(x_in, P_feats, P_raw, station_ids=st_ids, track_feats=track_feats)
            
            v_pred = v_pred.squeeze().cpu().numpy()
            v_true = v_true.cpu().numpy()
            x = x.cpu().numpy()
            t = item['t'].numpy()
            
            rmse = np.sqrt(np.mean((v_pred - v_true)**2))
            rmses.append((rmse, group_name, x, t, v_true, v_pred))

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x, v_true, 'k-', label='Truth')
            ax.plot(x, v_pred, 'r--', label='Pred')
            ax.set_title(f"{group_name} | RMSE: {rmse:.4f}")
            ax.legend()
            ax.grid(True)
            plt.savefig(os.path.join(plot_dir, f"{group_name}.png"))
            plt.close(fig)
            
    worst_dir = os.path.join(run_dir, 'plots', f'worst_epoch_{epoch:03d}')
    os.makedirs(worst_dir, exist_ok=True)
    rmses.sort(key=lambda x: x[0], reverse=True)
    for i in range(min(5, len(rmses))):
        rmse, name, x, t, v_true, v_pred = rmses[i]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, v_true, 'k-', label='Truth')
        ax.plot(x, v_pred, 'r--', label='Model')
        ax.set_title(f"WORST #{i+1}: {name} | RMSE: {rmse:.4f}")
        ax.legend()
        plt.savefig(os.path.join(worst_dir, f"worst_{i+1}_{name}.png"))
        plt.close(fig)

def train(config_path, recompute_track_stats=False, compute_track_stats_only=False):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    print(f"Loaded config from: {config_path}")
    print(f"Config Model: num_freqs={cfg['model'].get('num_freqs')}, freq_base={cfg['model'].get('freq_base')}")
        
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"run_{start_time}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Config saved later after modification
        
    logger = setup_logging(run_dir)
    logger.info(f"Starting run: {run_dir}")
    
    device = torch.device(cfg['experiment'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    data_cfg = cfg['data']
    h5_path = data_cfg['h5_path']
    splits_dir = data_cfg.get('splits_dir', None)
    use_track_feats = data_cfg.get('use_track_feats', False)
    
    # Train Dataset (Point)
    train_ds = SpeedProfileH5Dataset(
        h5_path=h5_path,
        splits_dir=splits_dir,
        split='train',
        mode='point',
        points_per_sample=data_cfg.get('points_per_traj', 128),
        force_endpoints_prob=data_cfg.get('force_endpoints_prob', 0.3),
        use_track_feats=use_track_feats,
        use_track_event_features=data_cfg.get('use_track_event_features', False),
        use_new_p_schema=data_cfg.get('use_new_p_schema', False)
    )
    
    val_ds_point = SpeedProfileH5Dataset(
        h5_path=h5_path,
        splits_dir=splits_dir,
        split='val',
        mode='point',
        points_per_sample=data_cfg.get('points_per_traj', 128),
        force_endpoints_prob=0.0,
        use_track_feats=use_track_feats,
        use_track_event_features=data_cfg.get('use_track_event_features', False),
        use_new_p_schema=data_cfg.get('use_new_p_schema', False)
    )
    
    val_ds_traj = SpeedProfileH5Dataset(
        h5_path=h5_path,
        splits_dir=splits_dir,
        split='val',
        mode='trajectory',
        use_track_feats=use_track_feats,
        use_track_event_features=data_cfg.get('use_track_event_features', False),
        use_new_p_schema=data_cfg.get('use_new_p_schema', False)
    )
    
    logger.info(f"Train size: {len(train_ds)}, Val size: {len(val_ds_point)}, TrackFeats={use_track_feats}")

    preprocessor = PPreprocessor()
    
    # Fast paths for continuous param schema avoiding interpolation deadlocks
    if cfg['data'].get('use_new_p_schema', False):
         logger.info("Extracting P_cont from Dataset efficiently for PPreprocessor...")
         p_cont_list = []
         indices = np.random.choice(len(train_ds), size=min(len(train_ds), 20000), replace=False)
         for idx in tqdm(indices, desc="Fetching P_cont"):
             # We just need p_cont. Let's bypass full TrackFeats rendering which is O(N_points * N_traj)
             # Directly fetching P is O(1).
             g_name = train_ds.group_names[idx]
             f = train_ds._get_file()
             P_arr = f[g_name]['P'][:]
             
             x0 = P_arr[0]
             xf = P_arr[1]
             L = xf - x0
             v0 = P_arr[2]
             vf = P_arr[3]
             dur = P_arr[6]
             M = P_arr[7]
             v_seg_max = train_ds.vsegmax_cache[idx] if hasattr(train_ds, 'vsegmax_cache') else 0.0
             p_cont = np.array([L, v0, vf, dur, M, v_seg_max], dtype=np.float32)
             p_cont_list.append(p_cont)
             
         stats = preprocessor.fit_cont(list(p_cont_list), use_robust_stats=True)
    else:
         stats = preprocessor.fit(dataset=train_ds, num_samples=20000)
         
    logger.info(f"Preprocessor stats: {stats}")
    preprocessor.save(os.path.join(run_dir, "norm_stats.json"))
    
    t_cfg = cfg['training']
    warmup_epochs = t_cfg.get('warmup_epochs', 0)
    ramp_epochs = t_cfg.get('ramp_epochs', 0)
    
    # 3.5 Stratified Sampling
    use_stratified = t_cfg.get('use_stratified_sampling', False)
    if use_stratified:
        logging.info("Computing sampler weights for Trajectory Stratified Sampling...")
        
        # Hard assertion: Must be trajectory-level items conceptually 
        # (Though train_ds here is `point` mode, the sampling is over TRAJECTORY indices).
        # We sample an index, and dataset __getitem__ extracts K points from it.
        # So sampling over dataset length (N trajectories) is correct.
        # But we must ensure the dataset actually represents trajectories, not fully flattened points.
        # `SpeedProfileH5Dataset` in point mode still has `len(ds) == num_trajectories`.
        
        lengths = np.zeros(len(train_ds))
        vmaxs = np.zeros(len(train_ds))
        
        for i in range(len(train_ds)):
            l_i, v_i = train_ds.get_traj_stats(i)
            lengths[i] = l_i
            vmaxs[i] = v_i
            
        logging.info(f"L Stats      => Min: {lengths.min():.1f}, Median: {np.median(lengths):.1f}, Max: {lengths.max():.1f}")
        logging.info(f"v_seg_max Stats => Min: {vmaxs.min():.1f}, Median: {np.median(vmaxs):.1f}, Max: {vmaxs.max():.1f}")
        
        if lengths.max() == 0.0 and vmaxs.max() == 0.0:
            raise RuntimeError("Trajectory stats caches are zero; mappings init skipped (check use_new_p_schema)")
        
        # Bins setup
        # L: [0, 3km), [3, 7km), [7, 15km), [15km, inf)
        l_bins = [0, 3000, 7000, 15000, float('inf')]
        # V: [0, 15), [15, 25), [25, 40+) m/s
        v_bins = [0, 15, 25, 40, float('inf')]
        
        l_idx = np.digitize(lengths, l_bins) - 1 # 0 to 3
        v_idx = np.digitize(vmaxs, v_bins) - 1   # 0 to 3 (actually wait, 40+ is bin 3, highest is 3)
        # digitize puts 40+ in index 3 if we use 4 edges?
        # edges = [0, 15, 25, 40]. len=4. indices: 1 for [0, 15), 2 for [15, 25), 3 for [25, 40), 4 for [40, inf).
        l_idx = np.clip(np.digitize(lengths, [3000, 7000, 15000]), 0, 3)
        v_idx = np.clip(np.digitize(vmaxs, [15, 25, 40]), 0, 3)
        
        # We have 4x4 = 16 bins max.
        bin_counts = {}
        for i in range(len(train_ds)):
            b = (l_idx[i], v_idx[i])
            bin_counts[b] = bin_counts.get(b, 0) + 1
            
        logging.info("Bin Counts (L_bin, V_bin): " + str(bin_counts))
        
        # Compute weights
        # w \propto 1 / count
        weights = np.zeros(len(train_ds))
        w_max = 1.0 / (0.01 * len(train_ds)) # Cap weight to roughly 100x minimum? Or just arbitrary cap
        # Let's cap at max(1/50, 1000/N) -> standard approach: min(w_max, 1/count)
        cap = 5.0 / max(1, min(bin_counts.values())) if len(bin_counts) > 0 else 1.0
        
        for i in range(len(train_ds)):
            b = (l_idx[i], v_idx[i])
            w_raw = 1.0 / max(1, bin_counts[b])
            weights[i] = min(cap, w_raw)
            
        # Normalize weights so mean=1 strictly not needed for WeightedRandomSampler but good for debugging
        weights = torch.tensor(weights, dtype=torch.float)
        
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
        logging.info("Initialized Stratified Trajectory Sampler.")
        
    else:
        # 50/50 Original/Segment Fallback
        logging.info("Computing sampler weights for 50/50 rebalancing (Legacy)...")
        targets = []
        for name in train_ds.group_names:
            targets.append(1 if "_seg_" in name else 0)
        
        targets = torch.tensor(targets, dtype=torch.long)
        n_seg = targets.sum().item()
        n_orig = len(targets) - n_seg
        
        logging.info(f"Dataset Composition: Originals={n_orig}, Segments={n_seg}")
        
        weights = torch.zeros_like(targets, dtype=torch.float)
        if n_orig > 0: weights[targets == 0] = 0.5 / n_orig
        if n_seg > 0: weights[targets == 1] = 0.5 / n_seg
        
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=data_cfg['batch_size'], 
        shuffle=False, # Sampler is mutually exclusive with shuffle
        sampler=sampler,
        num_workers=data_cfg.get('num_workers', 2),
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds_point, 
        batch_size=data_cfg['batch_size'], 
        shuffle=False, 
        num_workers=data_cfg.get('num_workers', 2),
        pin_memory=True
    )
    
    m_cfg = cfg['model']
    track_feat_dim = m_cfg.get('track_feat_dim', 3 if use_track_feats else 0)
    
    # Update config for reproducibility and loading
    m_cfg['extra_features_dim'] = 2
    
    # Save config (after modification)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    
    model = SpeedProfileGenerator(
        param_dim=m_cfg['param_dim'],
        hidden_dim=m_cfg['hidden_dim'],
        num_blocks=m_cfg['num_blocks'],
        track_feat_dim=track_feat_dim,
        num_stations=100, # Hardcode or config
        emb_dim=m_cfg.get('emb_dim', 4),
        num_freqs=m_cfg.get('num_freqs', 16),
        freq_base=m_cfg.get('freq_base', 10.0),
        extra_features_dim=2 # log_length + is_segment
    )
    
    # Assert freq_base range
    fb = m_cfg.get('freq_base', 10.0)
    if not (1.0 <= fb <= 100.0):
        print(f"WARNING: freq_base={fb} is outside suggested range [1.0, 100.0] for xi in [0,1].")
        # raise ValueError(f"freq_base must be in [1.0, 100.0], got {fb}") # Optional strictness
    else:
        print(f"freq_base={fb} is valid.")
    model.to(device)
    logger.info(f"Model initialized. ParamDim={m_cfg['param_dim']}, TrackDim={track_feat_dim}")
    
    t_cfg = cfg['training']
    # 6. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=float(t_cfg['lr']), weight_decay=float(t_cfg['weight_decay']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # 4. Loss
    criterion = SpeedProfileLossV3(
        lambda_peak=float(cfg['training'].get('lambda_peak', 0.0)),
        lambda_curv=float(cfg['training'].get('lambda_curv', 0.0)),
        lambda_acc=float(cfg['training'].get('lambda_acc', 0.0)),
        lambda_speed_limit=float(cfg['training'].get('lambda_speed_limit', 0.0)),
        lambda_nonneg=float(cfg['training'].get('lambda_nonneg', 0.0)),
        lambda_energy=float(cfg['training'].get('lambda_energy', 0.0)),
        lambda_power=float(cfg['training'].get('lambda_power', 0.0)),
        lambda_pmp=float(cfg['training'].get('lambda_pmp', 0.0)),
        limit_alpha=float(cfg['training'].get('limit_alpha', 2.0)),
        limit_beta=float(cfg['training'].get('limit_beta', 10.0)),
        limit_gamma=float(cfg['training'].get('limit_gamma', 0.85)),
        use_cruise_weighting=cfg['training'].get('use_cruise_weighting', False),
        alpha_cruise=cfg['training'].get('alpha_cruise', 1.0),
        accel_proxy=cfg['training'].get('accel_proxy', 'dvdt')
    )
    
    # 5. Track Stats (Compute on Train Point Loader)
    # Check if we need to recompute or load
    track_stats_path = os.path.join(run_dir, 'track_stats.json')
    if recompute_track_stats and os.path.exists(track_stats_path):
        logger.info(f"Recompute requested: Deleting existing {track_stats_path}")
        os.remove(track_stats_path)
        
    if os.path.exists(track_stats_path):
        logger.info(f"Loading track stats from {track_stats_path}")
        with open(track_stats_path, 'r') as f:
            track_stats = json.load(f)
    else:
        # We use the point loader because it covers the distribution well and is already shuffled/loaded.
        track_stats = compute_track_stats(train_loader, device=device)
        # Save stats
        with open(track_stats_path, 'w') as f:
            json.dump(track_stats, f, indent=2)
            
    logging.info(f"Track Stats: {track_stats}")

    if compute_track_stats_only:
        logger.info("Computed track stats. Exiting as --compute_track_stats_only is set.")
        return

    # 6. Hybrid Setup (Trajectory Loader)
    traj_batches_per_epoch = cfg['training'].get('traj_batches_per_epoch', 0)
    train_loader_traj = None
    if traj_batches_per_epoch > 0:
        logging.info("Initializing Trajectory Dataset for Hybrid Training...")
        # Re-use h5_path
        h5_path = cfg['data']['h5_path']
        if not os.path.isabs(h5_path) and not os.path.exists(h5_path):
             h5_path = os.path.abspath(h5_path)
             
        train_ds_traj = SpeedProfileH5Dataset(
            h5_path=h5_path,
            split='train',
            mode='trajectory',
            use_track_feats=cfg['data'].get('use_track_feats', False),
            splits_dir=cfg['data'].get('splits_dir'),
            originals_only=cfg['training'].get('traj_originals_only', False),
            use_track_event_features=cfg['data'].get('use_track_event_features', False),
            use_new_p_schema=cfg['data'].get('use_new_p_schema', False)
        )
        
        train_loader_traj = torch.utils.data.DataLoader(
            train_ds_traj,
            batch_size=cfg['training'].get('traj_batch_size', 8),
            shuffle=True,
            num_workers=cfg['data']['num_workers'],
            collate_fn=pad_trajectory_batch,
            pin_memory=True
        )
        train_iter_traj = iter(train_loader_traj)
        logging.info(f"Trajectory Loader ready. Batches/Epoch: {traj_batches_per_epoch}")

    # 7. Training Loop
    logging.info("Starting training...")
    global_step = 0
    
    # Log Header
    with open(os.path.join(run_dir, 'training_log.csv'), 'w') as f:
        f.write("epoch,lambda_scale,train_loss,train_rmse,val_loss,val_rmse,grad_norm,traj_loss,peak_loss,curv_loss,acc_loss,cruise_loss,lim_frac,lim_max,neg_frac,orig_frac_seen\n")
        
    def get_lambda_scale(ep):
        if warmup_epochs == 0 and ramp_epochs == 0: return 1.0
        if ep < warmup_epochs: return 0.0
        if ep < warmup_epochs + ramp_epochs:
            return (ep - warmup_epochs) / float(ramp_epochs)
        return 1.0

    total_epochs = t_cfg['epochs'] # Renamed from 'epochs' to 'total_epochs' for consistency with original code
    best_val_rmse = float('inf')
    early_stopping_patience = t_cfg.get('early_stopping_patience', 0)
    early_stopping_min_delta = t_cfg.get('early_stopping_min_delta', 0.0)
    patience_counter = 0
    best_es_score = float('inf')
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0) 
    
    for epoch in range(total_epochs): 
        model.train()
        # ... (training loop content is same) ...
        # ... (validation loop content is same) ...
        
        # [We can't replace the whole loop efficiently with replace_file_content if it's huge]
        # I need to target the setup lines and the end-of-loop lines separately or use multi_replace.
        # But this tool call is replacing a chunk.
        # The user instruction implies replacing the setup and the checkpoint block.
        # I should simply use multi_replace or carefully target the blocks.
        # Let's target the SETUP first.

        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_batches = 0
        
        # Traj stats accumulators
        train_traj_loss = 0.0
        train_peak_loss = 0.0
        train_curv_loss = 0.0
        train_acc_loss = 0.0
        train_cruise_loss = 0.0
        train_energy_loss = 0.0
        train_power_loss = 0.0
        train_pmp_loss = 0.0
        train_lim_os = 0.0
        
        traj_loss_sum = 0.0
        peak_loss_sum = 0.0
        curv_loss_sum = 0.0
        acc_loss_sum = 0.0
        cruise_loss_sum = 0.0
        energy_loss_sum = 0.0
        power_loss_sum = 0.0
        pmp_loss_sum = 0.0
        
        lim_frac_sum = 0.0
        lim_mean_sum = 0.0
        lim_max_epoch = 0.0
        
        os_rate_pred_sum = 0.0
        os_rate_gt_sum = 0.0
        
        valid_dx_ratio_sum = 0.0
        valid_d2_ratio_sum = 0.0
        p_mw_max_epoch = 0.0
        
        dx_min_sum = 0.0
        dx_med_sum = 0.0
        dx_max_sum = 0.0
        zero_dx_ratio_sum = 0.0
        
        neg_frac_sum = 0.0
        neg_min_epoch = 0.0
        v_pred_max_epoch = -1e9
        traj_count = 0
        
        # For logging original fraction seen
        orig_frac_seen = 0.0
        
        # Point Loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}") # Use total_epochs
        for batch_idx, batch in enumerate(pbar): # Added batch_idx
            # Point Batch
            # x: [B, 1], P: [B, 8], v: [B, 1]
            x = batch['xi'].to(device)
            v_true = batch['v'].to(device)
            P_raw = batch['P'].to(device)
            is_seg_batch = batch['is_segment'].to(device) # [B, 1]
            
            # Augment P
            # log_length = log(xf - x0)
            x0 = P_raw[:, 0]
            xf = P_raw[:, 1]
            length = torch.clamp(xf - x0, min=1e-6)
            log_length = torch.log(length).unsqueeze(-1) # [B, 1]
            
            # Preprocess using new schema
            P_cont = P_raw[:, :6]
            P_feats = preprocessor.transform_cont(P_cont)
            
            # Concatenate Augmented Features
            # P_feats: [B, 8 + 1 + 1] -> [B, 10]
            P_feats = torch.cat([P_feats, log_length, is_seg_batch], dim=-1)
            
            st_ids = batch.get('station_ids')
            if st_ids is not None: st_ids = st_ids.to(device)
            
            track_feats = None
            if 'track_feats' in batch:
                tf_raw = batch['track_feats'].to(device)
                track_feats = normalize_track_feats(tf_raw, track_stats, device)
            
            optimizer.zero_grad()
            
            # Forward Point
            v_pred = model.forward_xi(x, P_feats, P_raw, station_ids=st_ids, track_feats=track_feats) 
            
            loss = criterion.mse(v_pred, v_true).mean()
            
            # Metrics for logging
            seg_frac_batch = is_seg_batch.mean().item()
            orig_frac_seen += (1.0 - seg_frac_batch) * x.size(0)
            
            # Hybrid Trajectory Step
            if train_loader_traj and traj_batches_per_epoch > 0:
                # Run small number of traj batches interleaved? 
                # Or just run 1 traj batch for every K point batches?
                # User spec: "inner training loop... Run a limited number of trajectory batches"
                # Let's run (traj_batches_per_epoch / len(train_loader)) prob? 
                # Or easier: Run N traj steps evenly spaced.
                # Actually, simplest is to just run a fixed number of steps per epoch, 
                # maybe spread out or at end.
                # Let's do: if (batch_idx % K == 0) -> run traj step.
                # Calculating stride
                stride = len(train_loader) // max(1, traj_batches_per_epoch)
                if stride == 0: stride = 1
                
                if batch_idx % stride == 0 and traj_count < traj_batches_per_epoch: # Use batch_idx
                    try:
                        batch_traj = next(train_iter_traj)
                    except StopIteration:
                        train_iter_traj = iter(train_loader_traj)
                        batch_traj = next(train_iter_traj)
                        
                    # Traj Forward
                    t_t = batch_traj['t'].to(device)
                    x_t = batch_traj['x'].to(device)
                    v_t_true = batch_traj['v'].to(device)
                    P_t_raw = batch_traj['P'].to(device)
                    mask_t = batch_traj['mask'].to(device)
                    xi_t = batch_traj['xi'].to(device)
                    is_seg_batch_traj = batch_traj['is_segment'].to(device) # [B, 1]
                    
                    accel_t = None
                    if 'accel' in batch_traj:
                         accel_t = batch_traj['accel'].to(device)
                    
                    tf_t = None
                    if 'track_feats' in batch_traj:
                        tf_t_raw = batch_traj['track_feats'].to(device)
                        # Normalize [B, T, 3]
                        tf_t = normalize_track_feats(tf_t_raw, track_stats, device)
                        
                        # LIMIT_IDX = 1 (speed limit in m/s)
                        v_limit_raw = tf_t_raw[:, :, 1].unsqueeze(-1)
                    else:
                        v_limit_raw = None

                    # --- DEBUG: Batch 0 Diags ---
                    if epoch == 0 and batch_idx == 0 and tf_t is not None:
                         print("\n" + "="*50)
                         print("  DEBUG: TRAJ BATCH 0 DIAGNOSTICS")
                         print("="*50)
                         
                         # Check Raw SpdLimit
                         sl_raw = tf_t_raw[:, :, 1]
                         mask_bool = batch_traj.get('mask', None)
                         if mask_bool is None:
                             # Reconstruct mask from padding? 
                             # Collator usually pads with 0. 
                             # But let's look at `mask_t` defined later? 
                             # Ah, mask_t is defined at start of loop? No.
                             # Let's check where mask_t comes from. 
                             pass
                         
                         print(f"  [Raw SpdLimit] All: Mean={sl_raw.mean():.4f}, Min={sl_raw.min():.4f}, Max={sl_raw.max():.4f}")
                         
                         # Check Normalized SpdLimit
                         sl_norm = tf_t[:, :, 1]
                         print(f"  [Norm SpdLimit] All: Mean={sl_norm.mean():.4f}, Min={sl_norm.min():.4f}, Max={sl_norm.max():.4f}")
                         
                         # Check Channel Ordering
                         print("  [Sample Raw] First Traj, First 5 pts:")
                         print(tf_t_raw[0, :5, :].cpu().numpy())
                         
                         print("="*50 + "\n")
                        
                    # Augment P_t_raw
                    x0_t = P_t_raw[:, 0]
                    xf_t = P_t_raw[:, 1]
                    length_t = torch.clamp(xf_t - x0_t, min=1e-6)
                    log_length_t = torch.log(length_t).unsqueeze(-1) # [B, 1]
                    P_t_cont = P_t_raw[:, :6]
                    P_t_feats = preprocessor.transform_cont(P_t_cont)                       
                    P_t_feats = torch.cat([P_t_feats, log_length_t, is_seg_batch_traj], dim=-1) # [B, 8] or [B, 10]
                    
                    # We need to reshape for forward if model doesn't support [B, T] directly? 
                    # SpeedProfileGenerator `forward` expects [B, 1] usually.
                    # `forward_x` in model handles [B, T] or [B, 1]? 
                    # Let's check model. Usually need to flatten or use forward_x.
                    # Model has forward_x(x, P_feats, P_raw, track_feats).
                    
                    st_ids_t = batch_traj.get('station_ids', None)
                    if st_ids_t is not None: st_ids_t = st_ids_t.to(device)

                    # --- Enforce Monotonicity by Sorting by x ---
                    sort_idx = torch.argsort(x_t, dim=1)
                    
                    def apply_sort(tensor, idx):
                        if tensor is None: return None
                        if tensor.dim() == 2:
                            return torch.gather(tensor, 1, idx)
                        elif tensor.dim() == 3:
                            idx_exp = idx.unsqueeze(-1).expand(-1, -1, tensor.size(-1))
                            return torch.gather(tensor, 1, idx_exp)
                        return tensor

                    x_t = apply_sort(x_t, sort_idx)
                    
                    # V3-V4 Loss Aggregators
                    traj_loss_sum = 0.0
                    peak_loss_sum = 0.0
                    curv_loss_sum = 0.0
                    acc_loss_sum = 0.0
                    cruise_loss_sum = 0.0
                    energy_loss_sum = 0.0
                    power_loss_sum = 0.0
                    
                    lim_frac_sum, lim_mean_sum = 0.0, 0.0
                    lim_max_epoch = 0.0
                    
                    # Overspeed rates
                    os_rate_pred_sum = 0.0
                    os_rate_gt_sum = 0.0
                    
                    # dx diagnostics
                    dx_min_sum, dx_med_sum, dx_max_sum = 0.0, 0.0, 0.0
                    zero_dx_ratio_sum = 0.0
                    
                    neg_frac_sum = 0.0
                    neg_min_epoch = 0.0
                    v_pred_max_epoch = 0.0
                    traj_count = 0
                    
                    t_t = apply_sort(t_t, sort_idx)
                    v_t_true = apply_sort(v_t_true, sort_idx)
                    P_t_raw = P_t_raw
                    mask_t = apply_sort(mask_t, sort_idx)
                    xi_t = apply_sort(xi_t, sort_idx)
                    
                    if accel_t is not None:
                        accel_t = apply_sort(accel_t, sort_idx)
                        
                    if tf_t is not None:
                        tf_t = apply_sort(tf_t, sort_idx)
                        
                    if v_limit_raw is not None:
                        v_limit_raw = apply_sort(v_limit_raw, sort_idx)
                        
                        # --- Speed Limit Assertions (Physical) ---
                        if batch_idx % 100 == 0:
                            assert torch.isfinite(v_limit_raw).all(), "v_limit contains NaNs/Infs"
                            # v_limit range in m/s should be roughly 10-45
                            assert v_limit_raw.min() >= 0.0, f"v_limit has negatives: {v_limit_raw.min()}"
                            assert v_limit_raw.max() < 200.0, f"v_limit > 200 m/s: {v_limit_raw.max()}"
                            
                            # Log stats once per epoch (approx)
                            if batch_idx == 0:
                                print(f"\n[Speed Limit Stats] Mean: {v_limit_raw.mean():.2f} m/s, Min: {v_limit_raw.min():.2f}, Max: {v_limit_raw.max():.2f}")
                        
                    # Re-verify dx after sort (Debug/One-time)
                    if batch_idx == 0 and epoch == 0: 
                        print("\n  [DX Check - Post Sort]")
                        dx_check = x_t[:, 1:] - x_t[:, :-1]
                        print(f"  dx shape: {dx_check.shape}")
                          
                        # Check for non-positives
                        neg_count = (dx_check <= 0).sum().item()
                        zero_count = (dx_check == 0).sum().item()
                        print(f"  dx <= 0 count: {neg_count}")
                        print(f"  dx == 0 count: {zero_count}")
                          
                        if neg_count > 0:
                            # Find first bad index
                            bad_b, bad_t = torch.where(dx_check <= 0)
                            b_idx = bad_b[0].item()
                            t_idx = bad_t[0].item()
                            print(f"  First bad dx at batch {b_idx}, t {t_idx}")
                            print(f"  x[{b_idx}, {t_idx}:{t_idx+5}]: {x_t[b_idx, t_idx:t_idx+5].cpu().tolist()}")
                            print(f"  dx[{b_idx}, {t_idx}:{t_idx+5}]: {dx_check[b_idx, t_idx:t_idx+5].cpu().tolist()}")
                            print(f"  mask[{b_idx}, {t_idx}:{t_idx+5}]: {mask_t[b_idx, t_idx:t_idx+5].cpu().tolist()}")


                    v_t_pred = model.forward_x(x_t, P_t_feats, P_t_raw, station_ids=st_ids_t, track_feats=tf_t) # [B, T, 1]
                    
                    # Traj Losses
                    # 1. Cruise/MSE
                    w_loss_cruise, r_loss_cruise = criterion.compute_cruise_loss(
                        v_t_pred, v_t_true, accel_t, t_t, x_t, mask_t
                    )
                    
                    # 2. Peak
                    w_loss_peak, r_loss_peak = criterion.compute_peak_loss(v_t_pred, v_t_true, mask_t)
                    
                    # 3. Curvature (V4 Robust Non-Uniform)
                    w_loss_curv, r_loss_curv = criterion.compute_curv_loss(v_t_pred, x_t.unsqueeze(-1), mask_t)
                    
                    # 4. Acceleration
                    # Pass both t_t and x_t so loss can choose based on proxy
                    w_loss_acc, r_loss_acc = criterion.compute_accel_loss(v_t_pred, accel_t, t=t_t, x=x_t, mask=mask_t)

                    # --- VERIFICATION BLOCK (Run once) ---
                    if global_step == 0 and accel_t is not None:
                        print("\n" + "="*40)
                        print("  ACCEL LOSS VERIFICATION (Batch 0)")
                        print("="*40)
                        
                        # 1. Check Monotonicity of X
                        dx_t = x_t[:, 1:] - x_t[:, :-1]
                        min_dx = dx_t.min().item()
                        max_dx = dx_t.max().item()
                        neg_dx_count = (dx_t <= 0).sum().item()
                        print(f"  dx stats: min={min_dx:.6f}, max={max_dx:.6f}, non-positive count={neg_dx_count}")
                        
                        # 2. Check Ranges
                        print(f"  v_pred range: [{v_t_pred.min().item():.3f}, {v_t_pred.max().item():.3f}]")
                        if accel_t is not None:
                            print(f"  a_true range: [{accel_t.min().item():.3f}, {accel_t.max().item():.3f}]")
                            
                        # 3. Analytical Consistency Check (GT vs GT)
                        # Compute "a_from_vdvdx_GT" using v_t_true and x_t
                        v_gt = v_t_true
                        v_mid_gt = (v_gt[:, 1:] + v_gt[:, :-1]) / 2.0
                        dv_gt = v_gt[:, 1:] - v_gt[:, :-1]
                        
                        # Use same dx logic (clamped)
                        dx_safe = torch.clamp(dx_t, min=1e-6).unsqueeze(-1)
                        
                        a_calc_gt = v_mid_gt * (dv_gt / dx_safe)
                        
                        # Compare with accel_t[:, :-1]
                        a_ref = accel_t[:, :-1, :]
                        
                        # Mask
                        val_mask = mask_t[:, :-1] & mask_t[:, 1:]
                        val_mask = val_mask.unsqueeze(-1)
                        
                        diff_a = (a_calc_gt - a_ref) * val_mask
                        mse_a = torch.sum(diff_a**2) / torch.sum(val_mask).clamp(min=1)
                        rmse_a = torch.sqrt(mse_a).item()
                        
                        print(f"  Analytic Check (GT vdvdx vs a_true): RMSE = {rmse_a:.6f}")
                        print("  (If RMSE is small, vdvdx logic matches dataset physics.)")
                        
                        # 4. Mask Checks
                        total_pts = mask_t[:, :-1].numel()
                        valid_pts = val_mask.sum().item()
                        print(f"  Masked points utilization: {valid_pts}/{total_pts} ({valid_pts/total_pts:.1%})")
                        
                        # 5. Check NaNs in calc
                        if torch.isnan(a_calc_gt).any():
                             print("  WARNING: NaNs found in calculated limits!")
                        
                        print("="*40 + "\n")
                    # -------------------------------------

                    v4_cfg = {
                        "DX_MIN_M": cfg['training'].get('DX_MIN_M', 1e-3),
                        "DX_MAX_M": cfg['training'].get('DX_MAX_M', 500.0),
                        "d2v_clamp": cfg['training'].get('d2v_clamp', 5.0),
                        "power_A": cfg['training'].get('power_A', 200.0),
                        "power_B": cfg['training'].get('power_B', 5.0),
                        "power_C": cfg['training'].get('power_C', 0.1),
                        "power_M": cfg['training'].get('power_M', 4e5),
                        "power_mode": cfg['training'].get('power_mode', 'log1p')
                    }

                    # 4. Curvature (V4)
                    w_loss_curv, curv_metrics = criterion.compute_curv_loss_v4(v_t_pred, x_t.unsqueeze(-1), v_gt=v_t_true, cfg=v4_cfg)
                    
                    # 5. Speed Limit (V4)
                    w_loss_lim, lim_metrics = criterion.compute_speed_limit_loss_v4(
                        v_t_pred, v_limit_raw, v_gt=v_t_true,
                        mask=mask_t,
                        alpha=cfg['training'].get('limit_alpha', 2.0),
                        beta=cfg['training'].get('limit_beta', 10.0),
                        gamma=cfg['training'].get('limit_gamma', 0.85)
                    )

                    # 6. Energy and Power (V4 Stack)
                    w_loss_energy, energy_metrics = criterion.compute_energy_loss_v4(v_t_pred, x_t.unsqueeze(-1), cfg=v4_cfg)
                    w_loss_power, power_metrics = criterion.compute_power_loss_v4(v_t_pred, x_t.unsqueeze(-1), cfg=v4_cfg)
                    w_loss_pmp, pmp_metrics = criterion.compute_pmp_loss_v4(v_t_pred, x_t.unsqueeze(-1), cfg=v4_cfg)
                    
                    r_loss_energy = energy_metrics.get("EnergyL_raw", 0.0)
                    r_loss_power = power_metrics.get("PowerL_raw", 0.0)
                    r_loss_pmp = pmp_metrics.get("PMPL_raw", 0.0)
                    r_loss_lim = lim_metrics.get("LimMax_pred", 0.0)
                    
                    r_curv_raw = curv_metrics.get("CurvL_raw", 0.0) # Actually I didn't return CurvL_raw, let me check losses_v3

                    # 7. Non-Negativity
                    w_loss_neg, r_loss_neg = criterion.compute_nonneg_loss(v_t_pred, mask_t)
                    
                    # Apply Annealing Scale to Physics Losses
                    lambda_scale = get_lambda_scale(epoch)
                    
                    # Combine V4 Stack
                    # Supervised: cruise, peak, acc
                    # Physics: curv, limit, energy, power, pmp
                    loss_traj = w_loss_cruise + w_loss_peak + w_loss_acc + \
                                w_loss_neg + \
                                lambda_scale * (w_loss_curv + w_loss_lim + w_loss_energy + w_loss_power + w_loss_pmp)
                    
                    loss += loss_traj 
                    
                    # Accumulate for logs
                    traj_loss_sum += loss_traj.item()
                    peak_loss_sum += r_loss_peak.item()
                    curv_loss_sum += r_curv_raw
                    acc_loss_sum += r_loss_acc.item()
                    cruise_loss_sum += r_loss_cruise.item()
                    energy_loss_sum += r_loss_energy
                    power_loss_sum += r_loss_power
                    pmp_loss_sum += r_loss_pmp
                    lim_frac_sum += lim_metrics.get("LimFrac_pred", 0.0)
                    
                    # Physical Stats
                    os_rate_pred_sum += lim_metrics.get("LimFrac_pred", 0.0)
                    os_rate_gt_sum += lim_metrics.get("LimFrac_gt", 0.0)
                    
                    # Accumulate Ratios
                    valid_dx_ratio_sum += curv_metrics.get("valid_dx_ratio", 0.0)
                    valid_d2_ratio_sum += curv_metrics.get("valid_d2_ratio", 0.0)
                    p_mw_max_epoch = max(p_mw_max_epoch, power_metrics.get("P_MW_max", 0.0))
                    
                    with torch.no_grad():
                        # dx stats on sorted trajectory
                        dx_min_sum += curv_metrics.get("dx_min", 0.0)
                        dx_med_sum += curv_metrics.get("dx_median", 0.0)
                        dx_max_sum += curv_metrics.get("dx_max", 0.0)
                        zero_dx_ratio_sum += curv_metrics.get("zero_dx_ratio", 0.0)
                    
                    # Neg / Max Pred Stats
                    with torch.no_grad():
                        neg_pts = (v_t_pred < 0).float()
                        neg_frac = neg_pts.mean().item()
                        neg_min = v_t_pred.min().item()
                        v_pred_max = v_t_pred.max().item()
                    
                    # Speed Limit Stats
                    lim_frac_sum += lim_metrics.get("LimFrac_pred", 0.0)
                    lim_mean_sum += lim_metrics.get("LimMean_pred", 0.0)
                    lim_max_epoch = max(lim_max_epoch, lim_metrics.get("LimMax_pred", 0.0))
                    
                    # Neg Stats
                    neg_frac_sum += neg_frac
                    neg_min_epoch = min(neg_min_epoch, neg_min)
                    v_pred_max_epoch = max(v_pred_max_epoch, v_pred_max)
                    
                    traj_count += 1
            
            loss.backward()
            
            # Grad Clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss_sum += loss.item()
            train_mse_sum += criterion.mse(v_pred, v_true).mean().item()
            train_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
            global_step += 1
            
        # Epoch Stats
        avg_train_loss = train_loss_sum / train_batches
        avg_train_rmse = np.sqrt(train_mse_sum / train_batches)
        
        avg_energy_loss = train_energy_loss / train_batches if train_batches > 0 else 0.0
        avg_power_loss = train_power_loss / train_batches if train_batches > 0 else 0.0
        avg_pmp_loss = train_pmp_loss / train_batches if train_batches > 0 else 0.0
        
        avg_lim_frac = train_lim_os / train_batches if train_batches > 0 else 0.0
        
        # Traj Avg
        if traj_count > 0:
            avg_traj_loss = traj_loss_sum / traj_count
            avg_peak_loss = peak_loss_sum / traj_count
            avg_curv_loss = curv_loss_sum / traj_count
            avg_acc_loss = acc_loss_sum / traj_count
            avg_cruise_loss = cruise_loss_sum / traj_count
            avg_energy_loss = energy_loss_sum / traj_count
            avg_power_loss = power_loss_sum / traj_count
            avg_pmp_loss = pmp_loss_sum / traj_count
            
            avg_lim_frac = lim_frac_sum / traj_count
            avg_lim_mean = lim_mean_sum / traj_count
            
            avg_os_pred = os_rate_pred_sum / traj_count
            avg_os_gt = os_rate_gt_sum / traj_count
            
            avg_dx_min = dx_min_sum / traj_count
            avg_dx_med = dx_med_sum / traj_count
            avg_dx_max = dx_max_sum / traj_count
            avg_zero_dx = zero_dx_ratio_sum / traj_count
            
            avg_neg_frac = neg_frac_sum / traj_count
        else:
            avg_traj_loss = 0.0
            avg_peak_loss = 0.0
            avg_curv_loss = 0.0
            avg_acc_loss = 0.0
            avg_cruise_loss = 0.0
            avg_energy_loss = 0.0
            avg_power_loss = 0.0
            avg_pmp_loss = 0.0
            avg_lim_frac = 0.0
            train_lim_os = 0.0 # This was missing
            avg_lim_mean, lim_max_epoch = 0.0, 0.0
            avg_neg_frac, neg_min_epoch, v_pred_max_epoch = 0.0, 0.0, 0.0
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['xi'].to(device)
                v = batch['v'].to(device)
                P_raw = batch['P'].to(device)
                is_seg_batch = batch['is_segment'].to(device) # [B, 1]
                
                # Augment P
                x0 = P_raw[:, 0]
                xf = P_raw[:, 1]
                length = torch.clamp(xf - x0, min=1e-6)
                log_length = torch.log(length).unsqueeze(-1) # [B, 1]
                
                # Extract Continuous Physical Parameters for Preprocessor
                P_cont = P_raw[:, :6]
                P_feats = preprocessor.transform_cont(P_cont)
                P_feats = torch.cat([P_feats, log_length, is_seg_batch], dim=-1)
                
                track_feats = None
                if 'track_feats' in batch:
                    tf_raw = batch['track_feats'].to(device)
                    track_feats = normalize_track_feats(tf_raw, track_stats, device)
                
                st_ids = batch.get('station_ids', None)
                if st_ids is not None: st_ids = st_ids.to(device)
                
                v_pred = model.forward_xi(x, P_feats, P_raw, station_ids=st_ids, track_feats=track_feats)
                loss = criterion.mse(v_pred, v).mean()
                val_loss_sum += loss.item()
                val_batches += 1
                
        avg_val_loss = val_loss_sum / max(1, val_batches)
        avg_val_rmse = np.sqrt(avg_val_loss)
        
        avg_val_rmse = np.sqrt(avg_val_loss)
        
        # Grad Norm (approx from last step)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Log epoch
        # orig_frac_seen is accumulated over the epoch.
        # The total number of samples seen in the epoch is len(train_loader) * data_cfg['batch_size']
        # However, with WeightedRandomSampler, the effective number of samples can be different.
        # A more robust way is to sum the batch sizes.
        total_samples_processed_in_epoch = train_batches * data_cfg['batch_size']
        orig_frac_epoch = orig_frac_seen / total_samples_processed_in_epoch if total_samples_processed_in_epoch > 0 else 0.0
        
        l_scale = get_lambda_scale(epoch)
        logging.info(f"Epoch {epoch+1}: LScale={l_scale:.3f} TrainRMSE={avg_train_rmse:.4f} ValRMSE={avg_val_rmse:.4f} "
                     f"TrajL={avg_traj_loss:.1e} Peak={avg_peak_loss:.1e} Curv={avg_curv_loss:.1e} Energy={avg_energy_loss:.1e} Power={avg_power_loss:.1e} "
                     f"LimFrac_P={avg_lim_frac:.1%} LimFrac_GT={avg_os_gt:.1%} LimMax={lim_max_epoch:.2f} "
                     f"dx_med={avg_dx_med:.2f} dx_zero={avg_zero_dx:.1%} "
                     f"NegFrac={avg_neg_frac:.1%} VMax={v_pred_max_epoch:.2f} OrigFracSeen={orig_frac_epoch:.2%}")
        
        logging.info(f"  Weighted Loss Breakdown: "
                     f"Peak={avg_peak_loss * criterion.lambda_peak:.1e} "
                     f"Curv={avg_curv_loss * criterion.lambda_curv * l_scale:.1e} "
                     f"Energy={avg_energy_loss * criterion.lambda_energy * l_scale:.1e} "
                     f"Power={avg_power_loss * criterion.lambda_power * l_scale:.1e} "
                     f"Limit={avg_lim_frac * avg_lim_mean * criterion.lambda_speed_limit * l_scale:.1e} "
                     f"PMP={avg_pmp_loss * criterion.lambda_pmp * l_scale:.1e}")
        
        with open(os.path.join(run_dir, 'training_log.csv'), 'a') as f:
            f.write(f"{epoch+1},{l_scale:.4f},{avg_train_loss:.6f},{avg_train_rmse:.6f},{avg_val_loss:.6f},{avg_val_rmse:.6f},{total_norm:.4f},"
                    f"{avg_traj_loss:.6f},{avg_peak_loss:.6f},{avg_curv_loss:.6f},{avg_acc_loss:.6f},{avg_cruise_loss:.6f},"
                    f"{avg_lim_frac:.6f},{lim_max_epoch:.6f},{avg_neg_frac:.6f},{orig_frac_epoch:.6f}\n")
            
        # Checkpoint
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pt'))
            logging.info(f"New best model saved! RMSE: {best_val_rmse:.4f}")
            
        # Early Stopping
        if early_stopping_patience > 0:
            if avg_val_rmse < (best_es_score - early_stopping_min_delta):
                best_es_score = avg_val_rmse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping triggered. No improvement in {early_stopping_patience} epochs.")
                    break
            
        # Frequent plots? Maybe every 10 epochs
        if (epoch + 1) % 10 == 0:
             save_plots(model, val_ds_traj, preprocessor, run_dir, device, epoch + 1, track_stats=track_stats)

    logging.info("Training complete.")
    report_path = os.path.join(run_dir, "validation_report.md")
    with open(report_path, "w") as f:
        f.write(f"# Training Report\nBest RMSE: {best_val_rmse}\nFinal RMSE: {avg_val_rmse}\n") # Use best_val_rmse and avg_val_rmse
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--recompute_track_stats', action='store_true', help='Force recomputation of track statistics')
    parser.add_argument('--compute_track_stats_only', action='store_true', help='Compute track stats and exit')
    args = parser.parse_args()
    
    train(args.config, recompute_track_stats=args.recompute_track_stats, compute_track_stats_only=args.compute_track_stats_only)
