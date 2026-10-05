import pytest
import numpy as np
from speed_profile_generator.data.h5_dataset import SpeedProfileH5Dataset
import tempfile
import h5py
import os

@pytest.fixture
def mock_dataset_with_track():
    # Create a temporary HDF5 file with dummy data to pass init checks
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "mock_data.h5")
        with h5py.File(h5_path, 'w') as f:
            g = f.create_group("traj_1")
            g.create_dataset("P", data=[0.0, 1000.0, 0.0, 0.0, 1.0, 2.0, 100.0, 1e5]) # L=1000
            g.create_dataset("x", data=[0.0, 500.0, 1000.0])
        
        # We subclass to inject dummy track data easily
        class MockDataset(SpeedProfileH5Dataset):
            def __init__(self, **kwargs):
                self.h5_path = h5_path
                self.group_names = ["traj_1"]
                self.split = "train"
                self.mode = "trajectory"
                self.originals_only = False
                self.use_track_feats = True
                self.use_track_event_features = True
                self.event_horizon = 0.4
                self.event_grid_steps = 33
                
                # Mock loaded track data: [N, 2] x, val
                # Constant slope=0, limit=30, curv=0 except a drop at x=800
                x_track = np.linspace(0, 1500, 100)
                slopes = np.zeros_like(x_track)
                limits = np.ones_like(x_track) * 30.0
                limits[x_track >= 800] = 10.0 # Drop at 80% of L
                curvs = np.zeros_like(x_track)
                
                self.track_data = {
                    'slope': np.column_stack([x_track, slopes]),
                    'spdLimit': np.column_stack([x_track, limits]),
                    'curvature': np.column_stack([x_track, curvs])
                }
                
                self.use_new_p_schema = True
                
                # Arrays needed for cache init
                self.L_cache = np.zeros(len(self.group_names), dtype=np.float32)
                self.vsegmax_cache = np.zeros(len(self.group_names), dtype=np.float32)
                
                self._init_mappings_and_stats()
                
        yield MockDataset()

def test_track_lookahead_shapes_and_fallback(mock_dataset_with_track):
    ds = mock_dataset_with_track
    
    # 1. Test shape (21 base + 5 events = 26)
    x_phys = np.array([500.0]) # Middle of traj
    xi_phys = np.array([0.5])  # Middle of traj
    out = ds._interpolate_track_features(x_phys, x0=0.0, L=1000.0, xi_phys=xi_phys)
    
    assert out.shape == (1, 26), f"Expected shape (1, 26), got {out.shape}"
    
    # 2. Test Drop Finding and Fallback
    # At xi=0.5 (x=500), drop is at x=800 (xi=0.8). Delta = 0.3.
    # Event horizon is 0.4. It should FIND the drop.
    dist_drop = out[0, 21 + 2]
    has_drop = out[0, 21 + 3]
    assert has_drop == 1.0, "Expected to find drop"
    assert np.isclose(dist_drop, 0.3, atol=0.05), f"Expected distance ~0.3, got {dist_drop}"
    
    # Now at xi=0.1 (x=100), drop is at x=800 (xi=0.8). Delta = 0.7.
    # Event horizon is 0.4. It should NOT find the drop.
    out2 = ds._interpolate_track_features(np.array([100.0]), x0=0.0, L=1000.0, xi_phys=np.array([0.1]))
    has_drop2 = out2[0, 21+3]
    dist_drop2 = out2[0, 21+2]
    assert has_drop2 == 0.0, "Expected NOT to find drop outside horizon"
    # Fallback formula: min(0.4, 1.0 - 0.1) = 0.4
    assert np.isclose(dist_drop2, 0.4), f"Expected fallback 0.4, got {dist_drop2}"
    
    # Now at xi=0.9 (x=900), no drops further down.
    # Fallback formula: min(0.4, 1.0 - 0.9) = 0.1
    out3 = ds._interpolate_track_features(np.array([900.0]), x0=0.0, L=1000.0, xi_phys=np.array([0.9]))
    has_drop3 = out3[0, 21+3]
    dist_drop3 = out3[0, 21+2]
    assert has_drop3 == 0.0
    assert np.isclose(dist_drop3, 0.1), f"Expected fallback at boundary 0.1, got {dist_drop3}"
    
    # 3. Test Clamping Boundary Behavior
    # At xi = 1.0, dxi_base goes up to +0.7 => xi' = 1.7 -> clamped to 1.0.
    # So the +0.7 feature should equal the 0.0 feature (since both clamped to 1.0).
    out_bound = ds._interpolate_track_features(np.array([1000.0]), x0=0.0, L=1000.0, xi_phys=np.array([1.0]))
    
    # index of dxi=0.0 is 2. index of dxi=0.7 is 6.
    base_idx_0 = 2 * 3
    base_idx_07 = 6 * 3
    assert np.allclose(out_bound[0, base_idx_0:base_idx_0+3], out_bound[0, base_idx_07:base_idx_07+3]), "Clamping failed: +0.7 lookahead does not match 0.0 lookahead at right boundary"
