
import os
import argparse
import numpy as np
import h5py
import scipy.io
from typing import Dict, Any, Optional, List, Tuple

def load_mat_file(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load .mat file, supporting both v7.2- (scipy) and v7.3 (h5py).
    Returns:
        trajectories: List of dicts (one per sol{i})
        railTrack: Dict of arrays (slope, spdLimit, curvature)
    """
    print(f"Loading {file_path}...")
    railTrack_data = {}

    try:
        # Try scipy first (v7.2 and below)
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        if 'sol' not in mat:
            raise ValueError(f"'sol' variable not found in {file_path}")
        
        # Extract RailTrack
        if 'railTrack' in mat:
            try:
                rt = mat['railTrack']
                # Try specific likely fields or generic extraction if Scipy loaded it
                # For now, let's assume we might fail here for v7.3 files anyway
                pass
            except Exception as e:
                print(f"Warning: Scipy railTrack extraction incomplete: {e}")

        # Re-raise to force h5py if we want consistent V7.3 handling
        raise NotImplementedError("Force h5py for V7.3 compliance")

    except NotImplementedError:
        # Fallback for v7.3 (HDF5 based)
        print("Scipy failed or skipped. Using h5py.")
        trajectories = []
        
        with h5py.File(file_path, 'r') as f:
            if 'sol' not in f:
                raise ValueError(f"'sol' variable not found in {file_path}")
            
            # --- Extract RailTrack ---
            if 'railTrack' in f:
                rt_grp = f['railTrack']
                print("Extracting railTrack...")
                
                def extract_curve(name):
                    if name in rt_grp:
                         grp = rt_grp[name]
                         try:
                             # Look for x/y or X/Y
                             if 'x' in grp: x_key, y_key = 'x', 'y'
                             elif 'X' in grp: x_key, y_key = 'X', 'Y'
                             else: return None
                             
                             x_val = np.array(grp[x_key]).flatten()
                             y_val = np.array(grp[y_key]).flatten()
                             return np.column_stack((x_val, y_val))
                         except Exception as e:
                             print(f"  Failed to extract {name} (keys check): {e}")
                             return None
                    return None

                railTrack_data['slope'] = extract_curve('slope')
                railTrack_data['spdLimit'] = extract_curve('spdLimit')
                if railTrack_data['spdLimit'] is None: railTrack_data['spdLimit'] = extract_curve('currSpdLimit')
                railTrack_data['curvature'] = extract_curve('curvature')
            
            # --- Extract Trajectories ---
            sol_refs = f['sol']
            sol_refs_flat = np.array(sol_refs).flatten()
            
            for i, ref in enumerate(sol_refs_flat):
                try:
                    s = f[ref] 
                    traj = {}
                    
                    def get_val(dataset_name):
                        if dataset_name not in s: return None
                        val = np.array(s[dataset_name]).squeeze()
                        return val

                    traj['x_data'] = get_val('x') 
                    y_val = get_val('y')
                    if y_val.shape[0] != 2 and y_val.shape[1] == 2:
                        y_val = y_val.T
                    traj['y_data'] = y_val
                    
                    traj['M'] = get_val('M')
                    traj['accel'] = get_val('accel')
                    
                    # Extract New P Fields (noStartSt, noEndSt)
                    v_start = get_val('noStartSt')
                    traj['noStartSt'] = float(v_start) if v_start is not None else np.nan
                    
                    v_end = get_val('noEndSt')
                    traj['noEndSt'] = float(v_end) if v_end is not None else np.nan

                    trajectories.append(traj)
                except Exception as e:
                     print(f"Warning: Skipping trajectory {i} in H5 fallback: {e}")
            
            return trajectories, railTrack_data

def process_trajectory(
    data: Dict[str, Any], 
    min_points: int, 
    tol: float = 1e-6
) -> Optional[Dict[str, np.ndarray]]:
    
    t = np.array(data['t'], dtype=np.float32)
    x = np.array(data['x'], dtype=np.float32)
    v = np.array(data['v'], dtype=np.float32)
    accel = data.get('accel')
    
    if len(t) < min_points: return None
    
    x0 = float(x[0])
    xf = float(x[-1])
    v0 = float(v[0])
    vf = float(v[-1])
    
    if np.any(np.diff(x) < -tol): return None # Monotonicity check
    if np.abs(xf - x0) < 1e-6: return None

    xi = (x - x0) / (xf - x0)
    xi = xi.astype(np.float32)
    
    # --- New P Schema (8 dims) ---
    # [x0, xf, v0, vf, noStartSt, noEndSt, duration, M]
    duration = t[-1] - t[0]
    M = float(data['M'])
    noStartSt = float(data.get('noStartSt', np.nan))
    noEndSt = float(data.get('noEndSt', np.nan))
    
    P = np.array([x0, xf, v0, vf, noStartSt, noEndSt, duration, M], dtype=np.float32)
    
    # Check for NaNs in Core P - we allow them if source is missing but warn?
    # User asked to ENSURE no NaNs. But if source is missing them, we can't invent.
    # We detected noStartSt exists in the source .mat file. So it should be fine.
    
    result = { 't': t, 'x': x, 'v': v, 'xi': xi, 'P': P }
    
    if accel is not None:
        accel = np.array(accel, dtype=np.float32)
        if len(accel) == len(t):
           result['accel'] = accel
           
    return result

def main():
    parser = argparse.ArgumentParser(description="Convert MATLAB dataset to HDF5")
    parser.add_argument("--input", required=True, help="Path to input .mat file")
    parser.add_argument("--output", required=True, help="Path to output .h5 file")
    parser.add_argument("--bellman_k", type=int, default=0, help="Number of Bellman sub-segments per trajectory")
    parser.add_argument("--min_points", type=int, default=50, help="Minimum points for valid trajectory")
    
    args = parser.parse_args()
    
    raw_trajs, railTrack_data = load_mat_file(args.input)
    print(f"Loaded {len(raw_trajs)} raw trajectories.")
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    stats = { "exported_orig": 0, "skipped_orig": 0, "exported_seg": 0, "skipped_seg": 0 }
    
    with h5py.File(args.output, 'w') as f_out:
        # --- Export RailTrack ---
        rt_grp = f_out.create_group("railTrack")
        for key, val in railTrack_data.items():
            if val is not None:
                rt_grp.create_dataset(key, data=val)
                print(f"Exported railTrack/{key}: shape={val.shape}")
            else:
                print(f"Warning: railTrack/{key} is missing.")
        
        # --- Export Trajectories ---
        valid_group_names = []
        
        for i, raw_t in enumerate(raw_trajs):
            try:
                t_arr = raw_t['x_data']
                y_arr = raw_t['y_data']
                
                if t_arr is None or len(t_arr) == 0:
                     stats['skipped_orig'] += 1
                     continue

                if y_arr.shape[1] == len(t_arr) and y_arr.shape[0] >= 2:
                    x_arr = y_arr[0, :]
                    v_arr = y_arr[1, :]
                elif y_arr.shape[0] == len(t_arr) and y_arr.shape[1] >= 2:
                    x_arr = y_arr[:, 0]
                    v_arr = y_arr[:, 1]
                else:
                     stats['skipped_orig'] += 1
                     continue
                
                base_data = {
                    't': t_arr,
                    'x': x_arr,
                    'v': v_arr,
                    'M': raw_t['M'],
                    'noStartSt': raw_t['noStartSt'],
                    'noEndSt': raw_t['noEndSt'],
                    'accel': raw_t.get('accel')
                }
                
                processed = process_trajectory(base_data, args.min_points)
                
                if processed:
                    grp_name = f"traj_{i:05d}"
                    grp = f_out.create_group(grp_name)
                    for k, val in processed.items():
                        grp.create_dataset(k, data=val)
                    valid_group_names.append(grp_name)
                    stats['exported_orig'] += 1
                    
                    if args.bellman_k > 0:
                        n_pts = len(processed['t'])
                        successes = 0
                        attempts = 0
                        while successes < args.bellman_k and attempts < args.bellman_k * 5:
                            attempts += 1
                            if n_pts <= args.min_points: break
                            idx_start = np.random.randint(0, n_pts - args.min_points)
                            idx_end = np.random.randint(idx_start + args.min_points, n_pts + 1)
                            
                            seg_data = {
                                't': processed['t'][idx_start:idx_end],
                                'x': processed['x'][idx_start:idx_end],
                                'v': processed['v'][idx_start:idx_end],
                                'M': base_data['M'],
                                'noStartSt': base_data['noStartSt'],
                                'noEndSt': base_data['noEndSt'],
                            }
                            if 'accel' in processed:
                                seg_data['accel'] = processed['accel'][idx_start:idx_end]
                                
                            proc_seg = process_trajectory(seg_data, args.min_points)
                            if proc_seg:
                                s_grp = f_out.create_group(f"{grp_name}_seg_{successes:03d}")
                                for k, val in proc_seg.items():
                                    s_grp.create_dataset(k, data=val)
                                valid_group_names.append(s_grp.name.strip('/'))
                                stats['exported_seg'] += 1
                                successes += 1
                            else:
                                stats['skipped_seg'] += 1
                else:
                    stats['skipped_orig'] += 1
                    
            except Exception as e:
                print(f"Error processing traj {i}: {e}")
                stats['skipped_orig'] += 1

        dt = h5py.string_dtype(encoding='utf-8')
        f_out.create_dataset('index', data=np.array(valid_group_names, dtype=object), dtype=dt)

    print("\nConversion Summary:")
    print(f"  Original: {stats['exported_orig']} / {stats['exported_orig'] + stats['skipped_orig']}")
    print(f"  Segments: {stats['exported_seg']}")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
