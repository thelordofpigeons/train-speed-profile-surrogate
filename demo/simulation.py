import numpy as np
import math

class Simulator:
    """
    Time-domain simulator for a train following a speed profile.
    Simple point-mass physics with acceleration constraints.
    """
    def __init__(self, x0, v0, dt=0.1, a_max=0.8, a_min=-1.2, 
                 tau=2.0, j_max=0.8, kp=1.0, k_t=0.5,
                 xf=None, x_stop_window=100.0, x_eps=1.0, v_eps=0.2, gamma=1.5):
        self.dt = dt
        self.state = {
            't': 0.0,
            'x': float(x0),
            'v': float(v0),
            'a': 0.0,
            'v_cmd': float(v0)
        }
        self.a_max = a_max
        self.a_min = a_min
        
        # Control Params
        self.tau = tau      
        self.j_max = j_max  
        self.kp = kp        
        self.k_t = k_t # Schedule gain (m/s / s)
        
        # Terminal Stop Params
        self.xf = xf
        self.x_stop_window = x_stop_window # keeping for backward compatibility if needed, but we use dynamic now
        self.x_eps = x_eps
        self.v_eps = v_eps
        self.gamma = gamma
        self.gamma_out = gamma + 0.3 # Hysteresis
        self.terminal_mode = False
        self.stopped = False

        # History
        self.history = {k: [v] for k, v in self.state.items()}
        self.history['v_ref'] = [v0] 
        self.history['disturbed'] = [0]
        self.history['e_t'] = [0.0]
        self.history['v_des'] = [v0]
        
        # Plan Storage
        self.plan_x = None
        self.plan_v = None
        self.plan_t = None
        self.t_plan_start = 0.0

    @staticmethod
    def compute_t_ref(plan_x, plan_v, eps=0.1):
        """
        Compute time reference t(x) for a given speed profile v(x).
        t[0] = 0.
        """
        plan_x = np.array(plan_x)
        plan_v = np.array(plan_v)
        N = len(plan_x)
        t_ref = np.zeros(N)
        
        # Cumulative trapezoidal integration
        # dt = dx / v_avg
        dx = np.diff(plan_x)
        v_mid = 0.5 * (plan_v[:-1] + plan_v[1:])
        v_mid = np.maximum(v_mid, eps)
        
        dt_steps = dx / v_mid
        t_ref[1:] = np.cumsum(dt_steps)
        return t_ref

    def set_plan(self, plan_x, plan_v, t_start=0.0):
        """
        Update the active plan and precompute its time schedule.
        """
        self.plan_x = np.array(plan_x)
        self.plan_v = np.array(plan_v)
        self.t_plan_start = float(t_start)
        self.plan_t = self.compute_t_ref(self.plan_x, self.plan_v)
        # print(f"[Simulator] Plan updated. t_start={t_start:.2f}, Duration={self.plan_t[-1]:.2f}s")
    
    def step(self, v_target_override=None, disturbance=None):
        """
        Advance simulation by dt.
        If v_target_override is None, uses internal plan + schedule correction.
        """
        if self.stopped:
            return self.state

        t = self.state['t']
        x = self.state['x']
        v = self.state['v']
        a_prev = self.state['a']
        v_cmd_prev = self.state.get('v_cmd', v)
        
        # Debug / Info vars
        debug_info = {}
        
        # 0. Terminal Arrival / Clamp Logic
        if self.xf is not None:
             if x >= (self.xf - self.x_eps) and v <= self.v_eps:
                # Goal reached. Stop cleanly without spikes.
                self.stopped = True
                self.state = {
                    't': t + self.dt, 
                    'x': self.xf, 
                    'v': 0.0, 
                    'a': 0.0, 
                    'v_cmd': 0.0
                }
                self._log(0, 0, 0, 0)
                return self.state

        # 1. Base Constraints
        constraints = {
            'a_max': self.a_max,
            'a_min': self.a_min,
            'v_limit': float('inf')
        }
        
        # 2. Apply Disturbance
        is_active = 0
        if disturbance:
            constraints = disturbance.apply(t, self.state, constraints)
            if constraints.get('v_limit', float('inf')) < 1e9 or constraints.get('a_max', self.a_max) != self.a_max:
                is_active = 1

        # 3. Control Logic Reference Generation
        v_ref = 0.0
        e_t = 0.0
        dv_sched = 0.0
        
        if v_target_override is not None:
            # Legacy Mode
            v_ref = v_target_override
            v_des = v_ref
        elif self.plan_x is not None:
            # Schedule Control Mode
            # 3a. Interpolate Plan
            # Use lookahead for v_plan to ensure we don't get stuck at v=0 start
            # Consistent with legacy loop
            lookahead = 5.0 
            x_query = min(x + lookahead, self.plan_x[-1])
            v_plan = np.interp(x_query, self.plan_x, self.plan_v)
            t_ref_val = np.interp(x, self.plan_x, self.plan_t)
            
            # 3b. Time Error: e_t = (SimTime - PlanStart) - ScheduledTime(x)
            # Late => e_t > 0
            e_t = (t - self.t_plan_start) - t_ref_val
            
            # 3c. Schedule Correction
            # Gating: Don't accelerate if plan is trying to stop (v_plan < 0.5)
            v_stop_threshold = 0.5
            dv_max = 3.0
            
            if v_plan < v_stop_threshold:
                dv_sched = 0.0
            else:
                dv_sched = np.clip(self.k_t * e_t, -dv_max, dv_max)
            
            # 3d. Desired Speed
            # Cap at max plan speed (or explicit limit if we had one per x)
            # We use constraints['v_limit'] as hard safety, but for corrective headroom we look at plan max
            v_cap = np.max(self.plan_v) 
            v_des = np.clip(v_plan + dv_sched, 0, v_cap)
            
            v_ref = v_des
            
            debug_info.update({
                'v_plan': v_plan,
                't_ref_val': t_ref_val,
                'e_t': e_t,
                'dv_sched': dv_sched
            })
        else:
            # Fallback if no plan
            v_ref = v_cmd_prev
            v_des = v_ref
            
        # 3e. Lag Filter (v_ref -> v_cmd)
        # Improved discretization: alpha = 1 - exp(-dt/tau)
        if self.tau > 1e-4:
            alpha = 1.0 - math.exp(-self.dt / self.tau)
            v_cmd = v_cmd_prev + alpha * (v_ref - v_cmd_prev)
        else:
            v_cmd = v_ref
            
        # B. Effective Target (Safety Constraint)
        v_cmd_effective = min(v_cmd, constraints.get('v_limit', float('inf')))
        
        # C. Proportional Control
        a_req = self.kp * (v_cmd_effective - v)
        
        # D. ROBUST TERMINAL MODE (Dynamic with Hysteresis & Jerk Limiting)
        if self.xf is not None:
            dist_rem = self.xf - x
            # d_stop_required = v^2 / (2 * |a_min|)
            d_stop_req = (v**2) / (2 * abs(self.a_min) + 1e-6)
            
            # Activation / Deactivation with Hysteresis
            if not self.terminal_mode:
                 if dist_rem > 0 and dist_rem <= self.gamma * d_stop_req:
                     self.terminal_mode = True
                     print(f"[TERMINAL] ACTIVE: t={t:.2f}s, d_rem={dist_rem:.1f}m, d_req={d_stop_req:.1f}m")
            else:
                 # Check condition to leave terminal mode (hysteresis)
                 if dist_rem > self.gamma_out * d_stop_req or dist_rem <= 0:
                     self.terminal_mode = False
                     print(f"[TERMINAL] INACTIVE: t={t:.2f}s")

            if self.terminal_mode:
                # Dominant braking: a = -v^2 / (2d)
                # But constrained by physical limits
                a_brake = - (v**2) / (2 * dist_rem + 0.1)
                
                # We blend or take min, but respecting jerk limits happens in Step E.
                # Here we just constrain the TARGET.
                # The user requested: a_safe = min(a_req, a_brake)
                a_req = min(a_req, a_brake)
                
                # Check for arrival
                if x >= self.xf - self.x_eps and v <= self.v_eps:
                     self.stopped = True

        # E. Constraints & Jerk Limiting
        # Handle 'accel_override' from disturbance (e.g. StopDisturbance)
        if 'accel_override' in constraints:
            a_req = constraints['accel_override']
            
        a_min_final = constraints.get('a_min', self.a_min)
        a_max_final = constraints.get('a_max', self.a_max)
        
        a_target = np.clip(a_req, a_min_final, a_max_final)
        
        # Jerk limiting (ignored/bypassed in terminal if needed? No, let's keep for physical consistency)
        if self.j_max > 0:
            da_max = self.j_max * self.dt
            da = np.clip(a_target - a_prev, -da_max, da_max)
            a_cmd = a_prev + da
        else:
            a_cmd = a_target
            
        a = np.clip(a_cmd, a_min_final, a_max_final)
        
        # DEBUG: Check for Jerk Violation
        if self.state['t'] > 0:
            j_curr = abs(a - a_prev) / self.dt
            if j_curr > self.j_max + 0.1: # Tolerance
                 pass # print(f"[JERK SPIKE] t={t:.2f}: j={j_curr:.2f}, a_prev={a_prev:.2f}, a={a:.2f}, source={'Stopped' if v+a*self.dt < 0 else 'Dynamics'}")

        # 4. Integration
        v_next = v + a * self.dt
        if v_next < 0:
            # We overshot 0. Hard stop.
            # Calculate the acceleration needed to stop exactly (infinite jerk potential)
            a_stop_exact = (0 - v) / self.dt
            
            # Clamp this to be jerk-limited relative to a_prev to avoid spikes in plots
            if self.j_max > 0:
                da_max = self.j_max * self.dt
                da_req = a_stop_exact - a_prev
                da_limited = np.clip(da_req, -da_max, da_max)
                a_smooth = a_prev + da_limited
            else:
                a_smooth = a_stop_exact
                
            # Also respect absolute limits
            a = np.clip(a_smooth, a_min_final, a_max_final)
            
            # Force v=0 (Simulator artifact: we stop "now" even if 'a' implies a bit later)
            v_next = 0
            
        x_next = x + v * self.dt + 0.5 * a * self.dt**2
        t_next = t + self.dt
        
        # Update State
        self.state = {
            't': t_next, 'x': x_next, 'v': v_next, 'a': a, 
            'v_cmd': v_cmd
        }
        
        # Store debug info in state for external access if needed
        self.state['debug'] = debug_info
        
        self._log(v_ref, is_active, e_t, v_des)
        return self.state

    def _log(self, v_ref, is_active, e_t, v_des):
        for k, val in self.state.items():
            if k == 'debug': continue # Skip dicts
            if k in self.history:
                self.history[k].append(val)
            else:
                 self.history[k] = [val]
        self.history['v_ref'].append(v_ref)
        self.history['disturbed'].append(is_active)
        self.history['e_t'].append(e_t)
        self.history['v_des'].append(v_des)

    def get_history(self):
        return {k: np.array(v) for k, v in self.history.items()}
