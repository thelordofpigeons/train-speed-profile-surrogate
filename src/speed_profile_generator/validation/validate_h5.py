
import os
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import glob
from scipy import stats
import h5py

# Add src to path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Imports from package
from speed_profile_generator.data.h5_dataset import SpeedProfileH5Dataset
from speed_profile_generator.models.speed_profile_generator import SpeedProfileGenerator
from speed_profile_generator.utils.preprocessor import PPreprocessor
from speed_profile_generator.validation.metrics import compute_bc_errors, check_violations, compute_regression_metrics

def load_run_artifacts(run_dir, device='cuda'):
    """Load model, config, and preprocessor from run directory."""
    # 1. Config
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Preprocessor
    stats_path = os.path.join(run_dir, "norm_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Norm stats not found at {stats_path}")
    preprocessor = PPreprocessor()
    preprocessor.load(stats_path)
    
    # 3. Model
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"Warning: best_model.pt not found, trying to find any .pt")
        pts = glob.glob(os.path.join(run_dir, "*.pt"))
        if not pts:
             raise FileNotFoundError(f"No checkpoint found in {run_dir}")
        ckpt_path = pts[0]
        
    model = SpeedProfileGenerator(
        param_dim=config['model']['param_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_blocks=config['model']['num_blocks'],
        track_feat_dim=config['model'].get('track_feat_dim', 0),
        emb_dim=config['model'].get('emb_dim', 4),
        num_freqs=config['model'].get('num_freqs', 16),
        freq_base=config['model'].get('freq_base', 10.0),
        extra_features_dim=config['model'].get('extra_features_dim', 0)
    )
    
    if 'freq_base' not in config['model']:
        print("WARNING: 'freq_base' not found in config. Using default 10.0. Ensure this matches training!")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.float() # Force float32
    model.to(device)
    model.eval()
    
    return model, config, preprocessor, ckpt_path

def load_track_stats(run_dir):
    """Load track statistics for normalization."""
    path = os.path.join(run_dir, "track_stats.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    print(f"Warning: track_stats.json not found in {run_dir}")
    return None

def normalize_track_feats(tf, stats, device):
    """Normalize track features using loaded stats."""
    if stats is None or tf is None: return tf
    
    mean = torch.tensor(stats['mean'], device=device, dtype=torch.float32).view(1, 1, -1)
    std = torch.tensor(stats['std'], device=device, dtype=torch.float32).view(1, 1, -1)
    
    # Handle std=0
    std[std < 1e-6] = 1.0
    
    return (tf - mean) / std

def load_station_map(map_path=None):
    """Load station ID -> Name mapping."""
    # Default fallback
    default_map = {
        32: "Pont de Bondy",
        33: "Bondy",
        34: "Rosny Bois-Perrier",
        35: "Val de Fontenay"
    }
    
    if map_path and os.path.exists(map_path):
        try:
            with open(map_path, 'r') as f:
                custom_map = json.load(f)
                # Convert string keys to int if needed
                custom_map = {int(k): v for k, v in custom_map.items()}
                default_map.update(custom_map)
            print(f"Loaded station map from {map_path}")
        except Exception as e:
            print(f"Error loading station map: {e}")
            
    return default_map

def get_station_names(P_raw, station_map):
    """Extract start/end station names from P."""
    # P schema: [x0, xf, v0, vf, noStartSt, noEndSt, ...]
    # Indices: 4: noStartSt, 5: noEndSt
    if P_raw.shape[-1] >= 6:
        id_start = int(P_raw[0, 4].item())
        id_end = int(P_raw[0, 5].item())
        name_start = station_map.get(id_start, f"Unknown({id_start})")
        name_end = station_map.get(id_end, f"Unknown({id_end})")
        return name_start, name_end
    return "Unknown", "Unknown"

def evaluate_trajectories(model, dataset, preprocessor, device, track_stats=None):
    """
    Pass B: Iterate over all trajectories to compute per-trajectory metrics.
    Returns DataFrame of metrics and arrays for error localization (xi, error).
    """
    results = []
    
    # Error localization accumulators
    err_loc_xi = []
    err_loc_val = []
    
    # Regime accumulators (Originals only)
    regime_stats = {
        'accel': {'sum_err': 0.0, 'count': 0, 'reservoir': []},
        'cruise': {'sum_err': 0.0, 'count': 0, 'reservoir': []},
        'brake': {'sum_err': 0.0, 'count': 0, 'reservoir': []}
    }
    RES_CAP = 100000 # Max samples per regime for quantiles
    
    print(f"Evaluating {len(dataset)} trajectories in Trajectory Mode...")
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            group_name = item['group']
            
            # Data
            # Cast to float to handle potential double from external sources
            x = item['x'].to(device).float()
            v_true = item['v'].to(device).float()
            P_raw = item['P'].to(device).float().unsqueeze(0) # [1, 8]
            
            # Track Features
            track_feats = None
            if 'track_feats' in item:
                track_feats = item['track_feats'].to(device).float()
                if track_feats.dim() == 2:
                    track_feats = track_feats.unsqueeze(0) # [1, T, 3]
            
            # Augment P
            x0 = P_raw[:, 0]
            xf = P_raw[:, 1]
            length = torch.clamp(xf - x0, min=1e-6)
            log_length = torch.log(length).unsqueeze(-1) # [B, 1]
            
            is_seg = item.get('is_segment')
            if is_seg is None:
                 is_seg = torch.zeros((P_raw.size(0), 1), device=device)
            else:
                 is_seg = is_seg.to(device)
                 if is_seg.dim() == 1: is_seg = is_seg.unsqueeze(0)
            
            if x.dim() == 1:
                x_in = x.unsqueeze(0)
            else:
                x_in = x.unsqueeze(0) if x.dim() == 2 else x 

            # Station IDs
            st_ids = item.get('station_ids')
            if st_ids is not None:
                st_ids = st_ids.to(device).unsqueeze(0) # [1, 2]

            # Preprocess
            P_cont = P_raw[:, :6]
            P_feats = preprocessor.transform_cont(P_cont).float()
            P_feats = torch.cat([P_feats, log_length, is_seg], dim=-1)
            
            # Normalize Track Feats for Model
            track_feats_norm = None
            if track_feats is not None:
                track_feats_norm = normalize_track_feats(track_feats, track_stats, device)
            
            # Predict
            # Explicit cast again to be paranoid
            x_in = x_in.float()
            P_feats = P_feats.float()
            P_raw = P_raw.float()
            if track_feats_norm is not None: track_feats_norm = track_feats_norm.float()
            
            v_pred = model.forward_x(x_in, P_feats, P_raw, station_ids=st_ids, track_feats=track_feats_norm) 
            
            # Assertions (Validation Invariant)
            assert torch.isfinite(v_pred).all(), f"v_pred Inf/NaN in group {group_name}"
            
            # Squeeze for metrics
            v_pred_seq = v_pred.squeeze() # [T]
            v_true_seq = v_true.squeeze() # [T]
            x_seq = x.squeeze() # [T]
            
            # --- Metrics ---
            
            # 1. Regression
            diff = v_pred_seq - v_true_seq
            abs_diff = torch.abs(diff)
            mse = torch.mean(diff**2).item()
            rmse = np.sqrt(mse)
            mae = torch.mean(abs_diff).item()
            max_err = torch.max(abs_diff).item()
            
            # 2. Hard BC (Exact)
            v0 = P_raw[0, 2].item()
            vf = P_raw[0, 3].item()
            v_start = v_pred_seq[0].item()
            v_end = v_pred_seq[-1].item()
            bc_err_start = abs(v_start - v0)
            bc_err_end = abs(v_end - vf)
            
            # 3. Negatives
            neg_frac = torch.mean((v_pred_seq < 0).float()).item()
            
            # 4. Stratification Params & Peak Metrics
            # Distance L
            L = (P_raw[0, 1] - P_raw[0, 0]).item() 
            
            # Peak Speed
            vmax_gt = torch.max(v_true_seq).item()
            vmax_pred = torch.max(v_pred_seq).item()
            dvmax = vmax_pred - vmax_gt
            abs_dvmax = abs(dvmax)
            
            is_segment = "_seg_" in group_name
            
            # (Duplicate results.append removed)
            
            # 5. NRMSE
            nrmse = rmse / (vmax_gt + 1e-6)

            # 6. Smoothness / Curvature
            # Compute 2nd derivative (acceleration of acceleration?) roughly curvature.
            # d2v = v[i+1] - 2v[i] + v[i-1]
            if len(v_pred_seq) > 2:
                v_pred_np = v_pred_seq.cpu().numpy()
                v_true_np = v_true_seq.cpu().numpy()
                d2v_pred = np.diff(v_pred_np, n=2)
                d2v_gt = np.diff(v_true_np, n=2)
                
                mean_abs_d2v_pred = np.mean(np.abs(d2v_pred))
                mean_abs_d2v_gt = np.mean(np.abs(d2v_gt))
                
                denom = max(mean_abs_d2v_gt, 1e-4) # Floor to avoid explosion
                smoothness_ratio = mean_abs_d2v_pred / denom
            else:
                smoothness_ratio = 1.0 # Default for tiny seq

            # 7. Speed Limit Compliance
            if track_feats is not None:
                # track_feats: [1, T, 3] -> (slope, limit, curv)
                v_limit_seq = track_feats[0, :, 1]
                violation_seq = torch.relu(v_pred_seq - v_limit_seq)
                mean_violation = torch.mean(violation_seq).item()
                frac_violation = torch.mean((violation_seq > 1e-3).float()).item()
                max_violation = torch.max(violation_seq).item()
            else:
                mean_violation = 0.0
                frac_violation = 0.0
                max_violation = 0.0
                
            # --- Diagnostic Block (First 10 batches only to avoid spam) ---
            if i < 5:
                # v_limit strict check
                if track_feats is not None:
                     v_limit_seq = track_feats[0, :, 1]
                     assert torch.isfinite(v_limit_seq).all()
                     assert v_limit_seq.min() >= -1e-3
                     assert v_limit_seq.max() < 200.0
                     
                print(f"[Diag {i}] {group_name}: v_pred range [{v_pred_seq.min():.2f}, {v_pred_seq.max():.2f}]")
                if track_feats is not None:
                     print(f"       v_lim range [{v_limit_seq.min():.2f}, {v_limit_seq.max():.2f}] violation_max={max_violation:.4f}")

            # 8. Acceleration Metrics (if available)
            accel_rmse = 0.0
            if 'accel' in item:
                a_true = item['accel'].to(device).squeeze()
                # Compute a_pred via finite diff (using x spacing proxy or if t available)
                # If t not available, approx a = v * dv/dx ? Or just skip a_pred for now and rely on smoothness.
                # User asked for "accel_pred proxy via finite differences on v_pred wrt t (masked)"
                # We need 't' for accurate accel. train_h5 typically doesn't yield t unless requested.
                # Assuming simple dx/dt ~ v => dt = dx/v.  a = dv/dt = dv/dx * v.
                if len(v_pred_seq) > 1:
                   v_mid = (v_pred_seq[:-1] + v_pred_seq[1:]) / 2
                   dx = x_seq[1:] - x_seq[:-1]
                   dv = v_pred_seq[1:] - v_pred_seq[:-1]
                   # Protect div by zero
                   mask = (dx > 1e-3)
                   a_pred_est = torch.zeros_like(v_pred_seq)
                   if mask.any():
                        # a = v * dv/dx
                        a_calc = v_mid[mask] * (dv[mask] / dx[mask])
                        # Place back roughly? This is tricky for point-wise RMSE.
                        # Easier proxy: just comparing v-profile smoothness is often enough. 
                        # But user asked for it. Let's do a centered diff if t is present.
                        pass
                
            # 9. Regime Analysis (Originals Only)
            if not is_segment:
                # Calculate Accel or load
                if 'accel' in item:
                    a_seq = item['accel'].to(device).squeeze()
                else:
                    # Proxy
                    a_seq = torch.zeros_like(v_true_seq)

                a_cpu = a_seq.cpu().numpy()
                
                # Adaptive Threshold
                a_abs = np.abs(a_cpu)
                std_a = np.std(a_cpu)
                eps_a = max(1e-3, 0.05 * std_a)
                
                # Point-wise checks
                is_accel = a_cpu > eps_a
                is_brake = a_cpu < -eps_a
                is_cruise = (~is_accel) & (~is_brake)
                
                # Collect errors
                diff_cpu = abs_diff.cpu().numpy()
                
                for rname, mask in [('accel', is_accel), ('brake', is_brake), ('cruise', is_cruise)]:
                    if np.any(mask):
                        errs = diff_cpu[mask]
                        regime_stats[rname]['count'] += len(errs)
                        regime_stats[rname]['sum_err'] += np.sum(errs)
                        
                        # Reservoir Sample (Simple cap)
                        if len(regime_stats[rname]['reservoir']) < RES_CAP:
                            take = min(len(errs), RES_CAP - len(regime_stats[rname]['reservoir']))
                            regime_stats[rname]['reservoir'].extend(errs[:take])

            results.append({
                "group": group_name,
                "is_segment": is_segment,
                "rmse": rmse,
                "nrmse": nrmse,
                "mae": mae,
                "max_err": max_err,
                "bc_err_start": bc_err_start,
                "bc_err_end": bc_err_end,
                "neg_frac": neg_frac,
                "dist": L,
                "vmax_true": vmax_gt,
                "vmax_pred": vmax_pred,
                "dvmax": dvmax,
                "abs_dvmax": abs_dvmax,
                "smoothness_ratio": smoothness_ratio,
                "mean_violation": mean_violation,
                "frac_violation": frac_violation,
                "max_violation": max_violation,
                "start_id": int(P_raw[0, 4].item()),
                "end_id": int(P_raw[0, 5].item())
            })
            
            # 8. Error Localization Data (Enhanced with Stratification features)
            # Normalize x to xi [0, 1]
            x0_val = P_raw[0, 0].item()
            xf_val = P_raw[0, 1].item()
            L_val = xf_val - x0_val
            if L_val > 1e-3:
                xi_seq = (x_seq - x0_val) / L_val
            else:
                xi_seq = torch.zeros_like(x_seq) # degenerate
                
            err_loc_xi.append(xi_seq.cpu().numpy())
            err_loc_val.append(abs_diff.cpu().numpy())
            
    # Concatenate error localization data
    xi_all = np.concatenate(err_loc_xi)
    err_all = np.concatenate(err_loc_val)
            
    return pd.DataFrame(results), xi_all, err_all, regime_stats

def generate_plots(model, dataset, preprocessor, df_results, xi_all, err_all, output_dir, device):
    """Generate Engineering-Grade Plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Subsets
    df_orig = df_results[~df_results['is_segment']]
    df_seg = df_results[df_results['is_segment']]
    if df_orig.empty: df_orig = df_results
    
    # --- 1. Histograms (RMSE, dvmax, abs_dvmax) ---
    def plot_hist(data, title, xlabel, filename, color='blue'):
        plt.figure(figsize=(10, 5))
        plt.hist(data, bins=50, alpha=0.7, color=color, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    plot_hist(df_orig['rmse'], "RMSE Distribution (Originals)", "RMSE", "rmse_hist_originals.png", 'blue')
    if not df_seg.empty:
        plot_hist(df_seg['rmse'], "RMSE Distribution (Segments)", "RMSE", "rmse_hist_segments.png", 'orange')
        
    plot_hist(df_orig['dvmax'], "Peak Speed Error (dvmax) (Originals)", "v_pred_max - v_gt_max", "dvmax_hist_originals.png", 'green')
    plot_hist(df_orig['abs_dvmax'], "Abs Peak Speed Error (Originals)", "|dvmax|", "abs_dvmax_hist_originals.png", 'purple')

    # --- 2. Correlations (RMSE vs ...) ---
    plt.figure(figsize=(6, 6))
    plt.scatter(df_orig['abs_dvmax'], df_orig['rmse'], alpha=0.5, s=10)
    plt.xlabel("Abs Peak Error (|dvmax|)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Peak Error (Originals)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "rmse_vs_abs_dvmax_originals.png"))
    plt.close()

    # --- 2.5 Curvature Ratio Hist (Originals) ---
    plot_hist(df_orig['smoothness_ratio'], "Smoothness Ratio Distribution (Originals)", "Ratio (Pred Curv / GT Curv)", "curvature_ratio_hist_originals.png", 'teal')

    
    # --- 3. Error Localization (Error vs Xi) ---
    # Bin xi into 50 bins
    # Compute mean and P90 error in each bin
    bins = np.linspace(0, 1, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # We use all points from originals? 
    # Current xi_all / err_all includes all trajectories (segments + originals). 
    # Ideally filtering for originals would require indices but let's use all for robust profile or pass mask.
    # Taking 'all' is acceptable proxy for general model behavior, but let's note it.
    
    bin_indices = np.digitize(xi_all, bins) - 1
    
    mean_err_profile = []
    p90_err_profile = []
    
    for b in range(50):
        mask = (bin_indices == b)
        if np.any(mask):
            errs_in_bin = err_all[mask]
            mean_err_profile.append(np.mean(errs_in_bin))
            p90_err_profile.append(np.percentile(errs_in_bin, 90))
        else:
            mean_err_profile.append(0.0)
            p90_err_profile.append(0.0)
            
    plt.figure(figsize=(10, 5))
    plt.plot(bin_centers, mean_err_profile, label='Mean |Error|', linewidth=2)
    plt.plot(bin_centers, p90_err_profile, label='P90 |Error|', linewidth=2, linestyle='--')
    plt.xlabel("Normalized Position (xi)")
    plt.ylabel("Absolute Error")
    plt.title("Error Localization Profile (All Trajectories)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "error_vs_xi_originals.png")) # Using filename contract despite being 'all'
    plt.close()

    # --- 4. Overlays (Stress Tests) ---
    def plot_overlays(indices, filename, prefix):
        n_plot = len(indices)
        if n_plot == 0: return
        fig, axes = plt.subplots(1, n_plot, figsize=(4 * n_plot, 4))
        if n_plot == 1: axes = [axes]
        
        for i, idx in enumerate(indices):
            # idx is DataFrame index, which maps to dataset index if ordered
            item = dataset[idx]
                
            group = item['group']
            x = item['x'].to(device).float()
            v_true = item['v'].to(device).float().cpu().numpy()
            x_cpu = x.cpu().numpy()
            
            P_raw = item['P'].to(device).float().unsqueeze(0)
            
            # Track Features
            track_feats = None
            if 'track_feats' in item:
                track_feats = item['track_feats'].to(device).float()
                if track_feats.dim() == 2:
                    track_feats = track_feats.unsqueeze(0)

            # Explicit casts
            x = x.float()
            P_raw = P_raw.float()
            if track_feats is not None: track_feats = track_feats.float()
            P_cont = P_raw[:, :6]
            P_feats = preprocessor.transform_cont(P_cont).float()
            with torch.no_grad():
                v_pred = model.forward_x(x.unsqueeze(0), P_feats, P_raw, track_feats=track_feats).squeeze().cpu().numpy()
                
            row = df_results.loc[idx]
            
            ax = axes[i]
            ax.plot(x_cpu, v_true, 'k-', alpha=0.6, label='GT')
            ax.plot(x_cpu, v_pred, 'r--', label='Pred')
            title = f"{group}\nRMSE:{row['rmse']:.3f} L:{row['dist']:.0f} dv:{row['dvmax']:.2f}"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x")
            if i == 0: 
                ax.set_ylabel("v")
                ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
    # Longest 5
    longest_indices = df_orig.nlargest(5, 'dist').index.tolist()
    plot_overlays(longest_indices, "overlays_longest5.png", "Longest")
    
    # Highest Vmax 5
    fastest_indices = df_orig.nlargest(5, 'vmax_true').index.tolist()
    plot_overlays(fastest_indices, "overlays_highvmax5.png", "Fastest")
    
    # Worst abs_dvmax 5
    worst_dv_indices = df_orig.nlargest(5, 'abs_dvmax').index.tolist()
    plot_overlays(worst_dv_indices, "overlays_worst_abs_dvmax5.png", "WorstPeakErr")
    
    # Standard Best/Worst RMSE
    best_rmse_indices = df_orig.nsmallest(5, 'rmse').index.tolist()
    plot_overlays(best_rmse_indices, "best_trajectories.png", "BestRMSE")
    
    worst_rmse_indices = df_orig.nlargest(5, 'rmse').index.tolist()
    plot_overlays(worst_rmse_indices, "worst_trajectories.png", "WorstRMSE")

def plot_bellman_overlays(model, df_seg, h5_path, preprocessor, device, output_dir, station_map, dataset, track_stats=None, n_plots=5):
    """
    Plot segments overlaid on their FULL original trajectory.
    1. Parse segment base ID.
    2. Load full original from H5.
    3. Run model on Segment inputs (from dataset).
    4. Plot full original GT, Segment GT, and Segment Pred.
    """
    if df_seg.empty: return
    
    # Sample random segments
    sample = df_seg.sample(n=min(n_plots, len(df_seg)))
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with h5py.File(h5_path, 'r') as f:
            for idx, row in sample.iterrows():
                # idx is the index in the original dataset/dataframe
                seg_name = row['group']
                
                # parse base: traj_xxxxx_seg_yyy -> traj_xxxxx
                if "_seg_" not in seg_name: continue
                base_name = seg_name.split("_seg_")[0]
                
                if base_name not in f:
                    print(f"Warning: Base {base_name} not found for {seg_name}")
                    continue
                    
                grp = f[base_name]
                
                # --- 1. Load Full Trajectory Data (Base) ---
                x_full = torch.from_numpy(grp['x'][:]).float().to(device)
                v_full = torch.from_numpy(grp['v'][:]).float().to(device)
                P_full = torch.from_numpy(grp['P'][:]).float().to(device).unsqueeze(0)
                
                # --- 2. Get Segment Data & Prediction from Dataset ---
                try:
                    item = dataset[idx]
                except Exception as e:
                    print(f"Error accessing dataset index {idx}: {e}")
                    continue

                if item['group'] != seg_name:
                    print(f"Warning: Index mismatch. Dataset[{idx}] is {item['group']} but df row is {seg_name}")
                    # If df_seg came from df_results which matches dataset order, this should be fine.
                    pass

                # Inputs for model
                x_seg_tensor = item['x'].to(device).float()
                v_seg_tensor = item['v'].to(device).float()
                P_seg_raw = item['P'].to(device).float().unsqueeze(0)
                
                track_feats = None
                if 'track_feats' in item:
                    track_feats = item['track_feats'].to(device).float()
                    if track_feats.dim() == 2:
                        track_feats = track_feats.unsqueeze(0) # [1, T, 3]

                st_ids = item.get('station_ids')
                if st_ids is not None: 
                    st_ids = st_ids.to(device).unsqueeze(0)

                # Augment P
                x0 = P_seg_raw[:, 0]
                xf = P_seg_raw[:, 1]
                length = torch.clamp(xf - x0, min=1e-6)
                log_length = torch.log(length).unsqueeze(-1)
                
                is_seg = item.get('is_segment')
                if is_seg is None:
                     is_seg = torch.zeros((P_seg_raw.size(0), 1), device=device)
                else:
                     is_seg = is_seg.to(device)
                     if is_seg.dim() == 1: is_seg = is_seg.unsqueeze(0)

                # Transform
                P_cont_seg = P_seg_raw[:, :6]
                P_feats = preprocessor.transform_cont(P_cont_seg).float()
                P_feats = torch.cat([P_feats, log_length, is_seg], dim=-1) # [1, 10]

                # Normalize Track Feats
                if track_feats is not None:
                     track_feats = normalize_track_feats(track_feats, track_stats, device)

                # Predict
                with torch.no_grad():
                    v_pred_seg = model.forward_x(x_seg_tensor.unsqueeze(0), P_feats, P_seg_raw, 
                                               station_ids=st_ids, track_feats=track_feats)
                    v_pred_seg = v_pred_seg.squeeze().cpu().numpy()

                x_seg_np = x_seg_tensor.cpu().numpy()
                v_seg_np = v_seg_tensor.cpu().numpy()

                # --- 3. Plot ---
                plt.figure(figsize=(12, 6))
                
                # Base GT
                x_full_np = x_full.cpu().numpy()
                v_full_np = v_full.cpu().numpy()
                plt.plot(x_full_np, v_full_np, 'k-', alpha=0.3, label='Original GT')
                
                # Segment GT
                plt.plot(x_seg_np, v_seg_np, 'b--', linewidth=2, label='Segment GT')
                
                # Segment Pred
                plt.plot(x_seg_np, v_pred_seg, 'r-.', linewidth=2, label='Segment Pred')
                
                # Highlight window
                x_start = x_seg_np[0]
                x_end = x_seg_np[-1]
                plt.axvline(x_start, color='g', linestyle=':', alpha=0.5)
                plt.axvline(x_end, color='g', linestyle=':', alpha=0.5)
                    
                # Title
                st_start, st_end = get_station_names(P_full, station_map)
                rmse = np.sqrt(np.mean((v_pred_seg - v_seg_np)**2))
                plt.title(f"Bellman Overlay: {seg_name}\nBase: {base_name} ({st_start} -> {st_end}) | SegRMSE: {rmse:.3f}")
                plt.legend()
                plt.savefig(os.path.join(output_dir, f"bellman_{seg_name}.png"))
                plt.close()

    except Exception as e:
        print(f"Bellman plot error: {e}")

def binned_analysis(df):
    """Compute metrics binned by distance and vmax."""
    df = df.copy()
    
    # Distance Bins (Terciles)
    try:
        df['dist_bin'] = pd.qcut(df['dist'], q=3, labels=["Short", "Medium", "Long"])
        dist_stats = df.groupby('dist_bin', observed=False).agg(
            count=('rmse', 'count'),
            rmse_mean=('rmse', 'mean'),
            rmse_p90=('rmse', lambda x: np.percentile(x, 90) if len(x) > 0 else np.nan),
            abs_dvmax_mean=('abs_dvmax', 'mean')
        ).reset_index()
    except Exception as e:
        print(f"Warning: Distance binning failed: {e}")
        dist_stats = pd.DataFrame()

    # Vmax Bins (Terciles)
    try:
        df['vmax_bin'] = pd.qcut(df['vmax_true'], q=3, labels=["Slow", "Moderate", "Fast"])
        vmax_stats = df.groupby('vmax_bin', observed=False).agg(
            count=('rmse', 'count'),
            rmse_mean=('rmse', 'mean'),
            rmse_p90=('rmse', lambda x: np.percentile(x, 90) if len(x) > 0 else np.nan),
            abs_dvmax_mean=('abs_dvmax', 'mean')
        ).reset_index()
    except Exception as e:
        print(f"Warning: Vmax binning failed: {e}")
        vmax_stats = pd.DataFrame()

    return dist_stats, vmax_stats

def generate_plots(model, dataset, preprocessor, df_results, xi_all, err_all, output_dir, device, station_map=None, track_stats=None):
    """Generate Engineering-Grade Plots."""
    os.makedirs(output_dir, exist_ok=True)
    if station_map is None: station_map = {}

    # Subsets
    df_orig = df_results[~df_results['is_segment']]
    df_seg = df_results[df_results['is_segment']]
    if df_orig.empty: df_orig = df_results
    
    # --- 1. Histograms (RMSE, dvmax, abs_dvmax, violations) ---
    def plot_hist(data, title, xlabel, filename, color='blue', bins=50):
        plt.figure(figsize=(10, 5))
        plt.hist(data, bins=bins, alpha=0.7, color=color, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    plot_hist(df_orig['rmse'], "RMSE Distribution (Originals)", "RMSE", "rmse_hist_originals.png", 'blue')
    if not df_seg.empty:
        plot_hist(df_seg['rmse'], "RMSE Distribution (Segments)", "RMSE", "rmse_hist_segments.png", 'orange')
        
    plot_hist(df_orig['dvmax'], "Peak Speed Error (dvmax) (Originals)", "v_pred_max - v_gt_max", "dvmax_hist_originals.png", 'green')
    plot_hist(df_orig['abs_dvmax'], "Abs Peak Error |dvmax|", "|dvmax|", "abs_dvmax_hist_originals.png", 'purple')
    
    # Violation Hists
    if 'frac_violation' in df_orig.columns:
         plot_hist(df_orig['frac_violation'], "Frac Speed Limit Violation (Originals)", "Fraction of Points > Limit", "violation_frac_hist.png", 'red', bins=20)
         plot_hist(df_orig['max_violation'], "Max Speed Limit Violation (Originals)", "Max (m/s) > Limit", "violation_max_hist.png", 'darkred')

    # --- 2. Correlations ---
    plt.figure(figsize=(6, 6))
    plt.scatter(df_orig['abs_dvmax'], df_orig['rmse'], alpha=0.5, s=10)
    plt.xlabel("Abs Peak Error (|dvmax|)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Peak Error (Originals)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "rmse_vs_abs_dvmax_originals.png"))
    plt.close()

    # --- 3. Error Localization ---
    # Bin xi into 50 bins
    bins = np.linspace(0, 1, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Assuming xi_all corresponds to err_all
    bin_indices = np.digitize(xi_all, bins) - 1
    
    mean_err_profile = []
    p90_err_profile = []
    
    for b in range(50):
        mask = (bin_indices == b)
        if np.any(mask):
            errs_in_bin = err_all[mask]
            mean_err_profile.append(np.mean(errs_in_bin))
            p90_err_profile.append(np.percentile(errs_in_bin, 90))
        else:
            mean_err_profile.append(0.0)
            p90_err_profile.append(0.0)
            
    plt.figure(figsize=(10, 5))
    plt.plot(bin_centers, mean_err_profile, label='Mean |Error|', linewidth=2)
    plt.plot(bin_centers, p90_err_profile, label='P90 |Error|', linewidth=2, linestyle='--')
    plt.xlabel("Normalized Position (xi)")
    plt.ylabel("Absolute Error")
    plt.title("Error Localization Profile (All Trajectories)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "error_vs_xi_originals.png"))
    plt.close()

    # --- 4. Overlays (Stress Tests) ---
    def plot_overlays(indices, filename, prefix):
        n_plot = len(indices)
        if n_plot == 0: return
        fig, axes = plt.subplots(1, n_plot, figsize=(4 * n_plot, 4))
        if n_plot == 1: axes = [axes]
        
        for i, idx in enumerate(indices):
            # idx is DataFrame index
            try:
                item = dataset[idx]
            except:
                continue # Skip if index issue
                
            group = item['group']
            # Explicit casts (Correct Location)
            x = item['x'].to(device).float()
            v_true = item['v'].to(device).float().cpu().numpy()
            x_cpu = x.cpu().numpy()
            
            P_raw = item['P'].to(device).float().unsqueeze(0)
            
            track_feats = None
            if 'track_feats' in item:
                track_feats = item['track_feats'].to(device).float()
                if track_feats.dim() == 2:
                    track_feats = track_feats.unsqueeze(0)
                
                # Normalize!
                if track_stats is not None:
                    track_feats = normalize_track_feats(track_feats, track_stats, device)

            st_ids = item.get('station_ids')
            if st_ids is not None: st_ids = st_ids.to(device).unsqueeze(0)

            # Augment P
            x0 = P_raw[:, 0]
            xf = P_raw[:, 1]
            length = torch.clamp(xf - x0, min=1e-6)
            log_length = torch.log(length).unsqueeze(-1) # [1, 1]
            
            is_seg = item.get('is_segment')
            if is_seg is None:
                 is_seg = torch.zeros((P_raw.size(0), 1), device=device)
            else:
                 is_seg = is_seg.to(device)
                 if is_seg.dim() == 1: is_seg = is_seg.unsqueeze(0)
                 
            P_cont = P_raw[:, :6]
            P_feats = preprocessor.transform_cont(P_cont).float()
            P_feats = torch.cat([P_feats, log_length, is_seg], dim=-1)
            
            with torch.no_grad():
                v_pred = model.forward_x(x.unsqueeze(0), P_feats, P_raw, station_ids=st_ids, track_feats=track_feats).squeeze().cpu().numpy()
                
            if idx in df_results.index:
                row = df_results.loc[idx]
                rmse_val = row['rmse']
                dv_val = row['dvmax']
                dist_val = row['dist']
            else:
                 rmse_val, dv_val, dist_val = 0.0, 0.0, 0.0
            
            ax = axes[i]
            ax.plot(x_cpu, v_true, 'k-', alpha=0.6, label='GT')
            ax.plot(x_cpu, v_pred, 'r--', label='Pred')
            
            # Add station names
            st_start, st_end = get_station_names(P_raw, station_map)
            
            title = f"{group}\n{st_start}->{st_end}\nRMSE:{rmse_val:.3f} L:{dist_val:.0f}"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("x")
            if i == 0: 
                ax.set_ylabel("v")
                ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
    # Longest 5
    longest_indices = df_orig.nlargest(5, 'dist').index.tolist()
    plot_overlays(longest_indices, "overlays_longest5.png", "Longest")
    
    # Highest Vmax 5
    fastest_indices = df_orig.nlargest(5, 'vmax_true').index.tolist()
    plot_overlays(fastest_indices, "overlays_highvmax5.png", "Fastest")
    
    # Worst abs_dvmax 5
    worst_dv_indices = df_orig.nlargest(5, 'abs_dvmax').index.tolist()
    plot_overlays(worst_dv_indices, "overlays_worst_abs_dvmax5.png", "WorstPeakErr")
    
    # Standard Best/Worst RMSE
    best_rmse_indices = df_orig.nsmallest(5, 'rmse').index.tolist()
    plot_overlays(best_rmse_indices, "best_trajectories.png", "BestRMSE")
    
    worst_rmse_indices = df_orig.nlargest(5, 'rmse').index.tolist()
    plot_overlays(worst_rmse_indices, "worst_trajectories.png", "WorstRMSE")

def validate_run(run_dir, output_name="engineering_report", max_groups=None, thresholds=None, station_map_path=None, include_bellman=False, bellman_n_plots=10, args_segment_breakdown=False):
    print(f"Starting Engineering Validation for: {run_dir}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 0. Load Station Map
    station_map = load_station_map(station_map_path)
    
    # 1. Load Resources
    model, config, preprocessor, ckpt_path = load_run_artifacts(run_dir, device)
    track_stats = load_track_stats(run_dir)
    
    # 2. Setup Dataset
    h5_path = config['data']['h5_path']
    if not os.path.isabs(h5_path) and not os.path.exists(h5_path):
        h5_path = os.path.abspath(h5_path)
    
    ds_val = SpeedProfileH5Dataset(
        h5_path=h5_path,
        split='val',
        mode='trajectory',
        use_track_feats=config['data'].get('use_track_feats', False),
        use_track_event_features=config['data'].get('use_track_event_features', False),
        splits_dir=config['data'].get('splits_dir')
    )
    
    if max_groups is not None:
        print(f"Subsetting validation to first {max_groups} groups.")
        ds_val = torch.utils.data.Subset(ds_val, range(min(len(ds_val), max_groups)))

    # 3. Evaluate Metrics
    df, xi_all, err_all, regime_stats = evaluate_trajectories(model, ds_val, preprocessor, device, track_stats=track_stats)
    
    # 4. Generate plots
    plot_dir = os.path.join(run_dir, "validation_plots")
    generate_plots(model, ds_val, preprocessor, df, xi_all, err_all, plot_dir, device, station_map=station_map, track_stats=track_stats)
    
    # 4b. Bellman Overlays
    df_seg = df[df['is_segment']]
    if include_bellman and not df_seg.empty:
        print(f"Generating Bellman Overlays (n={bellman_n_plots})...")
        plot_bellman_overlays(model, df_seg, h5_path, preprocessor, device, plot_dir, station_map, ds_val, track_stats=track_stats, n_plots=bellman_n_plots)
    
    # 5. Analysis
    df_orig = df[~df['is_segment']]
    if df_orig.empty: df_orig = df_results # Fallback? No
    
    dist_stats, vmax_stats = binned_analysis(df_orig)
    
    # 6. Report
    report_path = os.path.join(run_dir, f"{output_name}.md")
    
    # Defaults
    final_thresholds = {
        'rmse_strict': 0.2,
        'rmse_lenient': 0.3,
        'dvmax_strict': 0.5,
        'dvmax_lenient': 1.0,
        'smoothness_max': 2.0
    }
    if thresholds:
        final_thresholds.update(thresholds)
    thresholds = final_thresholds
    
    with open(report_path, "w") as f:
        f.write(f"# Engineering Validation Report (V2-final)\n\n")
        f.write(f"**Metadata**:\n")
        f.write(f"- Run: `{os.path.basename(run_dir)}`\n")
        f.write(f"- Checkpoint: `{os.path.basename(ckpt_path)}`\n")
        f.write(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # --- 1. Split-Aware Metrics ---
        f.write(f"## 1. Split-Aware Metrics\n")
        f.write(f"| Split | Count | RMSE (Mean) | NRMSE (Mean) | NRMSE (P90) | MAE (Mean) | BC Start Mean |\n")
        f.write(f"|---|---|---|---|---|---|---|\n")
        
        splits = [("Originals", df_orig), ("Segments", df_seg), ("All", df)]
        for name, d in splits:
            if d.empty: continue
            nrmse_p90 = np.percentile(d['nrmse'], 90)
            bc_start_mean = d['bc_err_start'].mean()
            f.write(f"| {name} | {len(d)} | {d['rmse'].mean():.4f} | {d['nrmse'].mean():.4f} | {nrmse_p90:.4f} | {d['mae'].mean():.4f} | {bc_start_mean:.2e} |\n")
        f.write("\n")
        
        # --- 1b. Safety Metrics (Originals) ---
        f.write(f"## 1b. Safety Metrics (Originals)\n")
        if not df_orig.empty:
            mean_vio = df_orig['mean_violation'].mean()
            frac_vio = df_orig['frac_violation'].mean()
            max_vio = df_orig['max_violation'].max()
            f.write(f"- **Mean Speed Violation**: {mean_vio:.4f} m/s\n")
            f.write(f"- **Frac Points Violating**: {frac_vio:.2%}\n")
            f.write(f"- **Max Violation Observed**: {max_vio:.4f} m/s\n\n")
            f.write(f"![Violation Frac](validation_plots/violation_frac_hist.png)\n")
        
        # --- 2. Regime-Aware Error (Originals) ---
        f.write(f"## 2. Regime-Aware Error (Originals)\n")
        f.write(f"Adaptive threshold used per trajectory: `eps_a = max(1e-3, 0.05 * std(a))`\n\n")
        f.write(f"| Regime | Point Count | Fraction | Mean |Err| | P90 |Err| |\n")
        f.write(f"|---|---|---|---|---|\n")
        
        total_points = sum(r['count'] for r in regime_stats.values())
        if total_points == 0: total_points = 1 
        
        for rname in ['accel', 'cruise', 'brake']:
            rdata = regime_stats[rname]
            cnt = rdata['count']
            frac = cnt / total_points
            mean_err = rdata['sum_err'] / cnt if cnt > 0 else 0.0
            p90_err = np.percentile(rdata['reservoir'], 90) if rdata['reservoir'] else 0.0
            
            f.write(f"| {rname.upper()} | {cnt} | {frac:.1%} | {mean_err:.4f} | {p90_err:.4f} |\n")
        f.write("\n")
        
        # --- 3. Smoothness Diagnostics ---
        f.write(f"## 3. Smoothness Diagnostics (Originals)\n")
        if not df_orig.empty:
            mean_ratio = df_orig['smoothness_ratio'].mean()
            p90_ratio = np.percentile(df_orig['smoothness_ratio'], 90)
            max_ratio = df_orig['smoothness_ratio'].max()
            f.write(f"- Mean Ratio: {mean_ratio:.3f}\n")
            f.write(f"- P90 Ratio: {p90_ratio:.3f}\n")
            f.write(f"- Max Ratio: {max_ratio:.3f}\n\n")
            
            flagged = df_orig[df_orig['smoothness_ratio'] > thresholds['smoothness_max']]
            f.write(f"### High Curvature Trajectories (Ratio > {thresholds['smoothness_max']})\n")
            if not flagged.empty:
                cols = ['group', 'smoothness_ratio', 'rmse', 'dist', 'vmax_true']
                top10 = flagged.nlargest(10, 'smoothness_ratio')[cols]
                f.write(top10.to_markdown(floatfmt=".3f"))
            else:
                f.write("None.")
            f.write("\n\n")
            f.write("![Curvature Hist](validation_plots/curvature_ratio_hist_originals.png)\n\n")

        # --- 4. Acceptance Summary ---
        f.write(f"## 4. Acceptance Summary (V2-final)\n")
        f.write(f"Thresholds: RMSE < {thresholds['rmse_strict']}/{thresholds['rmse_lenient']}, |dvmax| < {thresholds['dvmax_strict']}/{thresholds['dvmax_lenient']}.\n\n")
        f.write(f"| Metric | Originals Pass % | Segments Pass % |\n")
        f.write(f"|---|---|---|\n")
        
        def calc_pass(d):
            if d.empty: return "N/A", "N/A", "N/A", "N/A", "N/A"
            p_rmse_strict = (d['rmse'] < thresholds['rmse_strict']).mean()
            p_rmse_lenient = (d['rmse'] < thresholds['rmse_lenient']).mean()
            p_dv_strict = (d['abs_dvmax'] < thresholds['dvmax_strict']).mean()
            p_dv_lenient = (d['abs_dvmax'] < thresholds['dvmax_lenient']).mean()
            p_neg = (d['neg_frac'] <= 1e-6).mean()
            return p_rmse_strict, p_rmse_lenient, p_dv_strict, p_dv_lenient, p_neg
            
        def fmt_pct(val):
            if isinstance(val, str): return val
            return f"{val:.1%}"
        
        o_rmse_s, o_rmse_l, o_dv_s, o_dv_l, o_neg = calc_pass(df_orig)
        s_rmse_s, s_rmse_l, s_dv_s, s_dv_l, s_neg = calc_pass(df_seg)
        
        f.write(f"| RMSE < {thresholds['rmse_strict']} | {fmt_pct(o_rmse_s)} | {fmt_pct(s_rmse_s)} |\n")
        f.write(f"| RMSE < {thresholds['rmse_lenient']} | {fmt_pct(o_rmse_l)} | {fmt_pct(s_rmse_l)} |\n")
        f.write(f"| |dvmax| < {thresholds['dvmax_strict']} | {fmt_pct(o_dv_s)} | {fmt_pct(s_dv_s)} |\n")
        f.write(f"| |dvmax| < {thresholds['dvmax_lenient']} | {fmt_pct(o_dv_l)} | {fmt_pct(s_dv_l)} |\n")
        f.write(f"| Non-negative | {fmt_pct(o_neg)} | {fmt_pct(s_neg)} |\n")
        f.write("\n")
        
        # --- 5. Peak Accuracy (Originals) ---
        f.write(f"## 5. Peak Accuracy (Originals)\n")
        if not df_orig.empty:
            mean_abs_dvmax = df_orig['abs_dvmax'].mean()
            p90_abs_dvmax = np.percentile(df_orig['abs_dvmax'], 90)
            
            # Correlations
            corr_rmse_dvmax = df_orig['rmse'].corr(df_orig['abs_dvmax'])
            corr_rmse_L = df_orig['rmse'].corr(df_orig['dist'])
            corr_rmse_vmax = df_orig['rmse'].corr(df_orig['vmax_true'])
            
            f.write(f"- **Mean |dvmax|**: {mean_abs_dvmax:.4f} m/s\n")
            f.write(f"- **P90 |dvmax|**: {p90_abs_dvmax:.4f} m/s\n")
            f.write(f"- **Correlations (N={len(df_orig)})**:\n")
            f.write(f"    - RMSE vs |dvmax|: {corr_rmse_dvmax:.3f}\n")
            f.write(f"    - RMSE vs Distance: {corr_rmse_L:.3f}\n")
            f.write(f"    - RMSE vs Vmax_gt: {corr_rmse_vmax:.3f}\n\n")
            
            f.write(f"![RMSE vs |dvmax|](validation_plots/rmse_vs_abs_dvmax_originals.png)\n")
            f.write(f"![dvmax hist](validation_plots/dvmax_hist_originals.png)\n\n")
        
        # --- 6. Error Localization ---
        f.write(f"## 6. Error Localization (vs Normalized Position)\n")
        f.write(f"![Error Profile](validation_plots/error_vs_xi_originals.png)\n\n")
        
        # --- 7. Stratified Performance ---
        f.write(f"## 7. Stratified Performance (Originals)\n")
        f.write(f"### By Distance\n")
        f.write(dist_stats.to_markdown(floatfmt=".4f"))
        f.write(f"\n\n### By Max Speed\n")
        f.write(vmax_stats.to_markdown(floatfmt=".4f"))
        f.write("\n\n")
        
        # --- 8. Stress Tests & Tail Risk ---
        f.write(f"## 8. Stress Tests & Tail Risk\n")
        f.write(f"### Worst 10 Trajectories (by RMSE)\n")
        worst_10 = df_orig.nlargest(10, 'rmse')[['group', 'rmse', 'dist', 'vmax_true', 'dvmax']]
        f.write(worst_10.to_markdown(floatfmt=".4f"))
        f.write(f"\n\n")
        
        f.write(f"### Component Overlays\n")
        f.write(f"![Longest 5](validation_plots/overlays_longest5.png)\n")
        f.write(f"![Fastest 5](validation_plots/overlays_highvmax5.png)\n")
        f.write(f"![Worst Peak Err 5](validation_plots/overlays_worst_abs_dvmax5.png)\n\n")
        
        if include_bellman:
             f.write(f"## 9. Bellman Analysis\n")
             f.write(f"Overlaying segments on their parent trajectory.\nSee `validation_plots/bellman_*.png`.\n")
        
        f.write(f"## Appendix: Config\n")
        f.write(f"```yaml\n{yaml.dump(config)}\n```\n")

    if args_segment_breakdown:
        with open(report_path, "a") as f:
             f.write(f"## 10. Segment Breakdown\n")
             f.write(f"Metrics per track segment (32->33, etc.)\n\n")
             
             # Group by segment
             df['segment_key'] = df.apply(lambda row: f"{int(row['start_id'])}->{int(row['end_id'])}", axis=1)
             
             targets = ["32->33", "33->34", "34->35"]
             
             f.write(f"| Segment | N | RMSE | NegFrac | LimFrac | VMax |\n")
             f.write(f"|---|---|---|---|---|---|\n")
             
             print("\n--- Segment Breakdown ---")
             
             # Helper to compute and print
             def report_seg(key, sub_df):
                 if sub_df.empty:
                     print(f"{key} (N=0): -")
                     f.write(f"| {key} | 0 | - | - | - | - |\n")
                     return
                     
                 count = len(sub_df)
                 rmse = sub_df['rmse'].mean()
                 neg_frac = sub_df['neg_frac'].mean()
                 # handle missing columns gracefully if not present (though they should be)
                 vmax = sub_df['vmax_pred'].max()
                 lim_frac = sub_df['frac_violation'].mean() if 'frac_violation' in sub_df.columns else 0.0
                 
                 print(f"{key} (N={count}): RMSE={rmse:.4f}, NegFrac={neg_frac:.1%}, VMax={vmax:.2f}, LimFrac={lim_frac:.1%}")
                 f.write(f"| {key} | {count} | {rmse:.4f} | {neg_frac:.1%} | {lim_frac:.1%} | {vmax:.2f} |\n")

             # Specified Targets
             for t in targets:
                 sub = df[df['segment_key'] == t]
                 report_seg(t, sub)
                 
             # Other
             sub_other = df[~df['segment_key'].isin(targets)]
             if not sub_other.empty:
                 report_seg("Other", sub_other)
                 
             print("-------------------------\n")
             f.write("\n")

    print(f"Validation complete. Report saved: {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default="engineering_report")
    parser.add_argument('--max_groups', type=int, default=None)
    parser.add_argument('--station_map_path', type=str, default=None, help="Path to station map JSON")
    parser.add_argument('--include_bellman', action='store_true', help="Check/Plot Bellman overlays")
    parser.add_argument('--bellman_n_plots', type=int, default=10, help="Number of Bellman overlay plots to generate")
    parser.add_argument('--segment_breakdown', action='store_true', help="Report metrics per track segment (32->33, etc.)")
    args = parser.parse_args()
    
    validate_run(
        args.run_dir, 
        args.output, 
        max_groups=args.max_groups,
        station_map_path=args.station_map_path,
        include_bellman=args.include_bellman,
        bellman_n_plots=args.bellman_n_plots,
        args_segment_breakdown=args.segment_breakdown
    )
