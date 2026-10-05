from abc import ABC, abstractmethod
import numpy as np

class Disturbance(ABC):
    """Abstract base class for disturbances."""
    def __init__(self, t_start):
        self.t_start = t_start
        self.triggered = False

    @abstractmethod
    def apply(self, t, state, constraints):
        """
        Modify state or constraints based on disturbance.
        state: dict(t, x, v, a)
        constraints: dict(v_max, a_max, a_min)
        Returns: modified constraints (if any)
        """
        pass

    def check_trigger(self, t):
        if not self.triggered and t >= self.t_start:
            self.triggered = True
            return True
        return False

class StopDisturbance(Disturbance):
    """
    Forces a physics-based stop:
    1. Braking: Apply constant decel until v=0.
    2. Holding: Hold v=0 for hold_duration.
    3. Release: Return control to driver.
    """
    def __init__(self, t_start, duration=10.0, braking_accel=-1.2):
        super().__init__(t_start)
        self.hold_duration = duration # Time to HOLD at stop
        self.braking_accel = braking_accel
        
        # State
        self.state_name = "IDLE" # IDLE, BRAKING, HOLDING, RELEASED
        self.t_hold_start = None
        
    def apply(self, t, state, constraints):
        v_current = state['v']
        
        # Trigger Logic
        if self.state_name == "IDLE":
            if t >= self.t_start:
                self.state_name = "BRAKING"
                print(f"[STOP] Triggered at t={t:.2f}, v={v_current:.2f}. Braking...")
                
        if self.state_name == "BRAKING":
            # Override acceleration constraints to force braking
            # We want a exact = braking_accel
            # But we want to RESPECT JERK LIMITS.
            # So we don't clamp a_max/a_min directly to the target.
            # Instead we request an override target, and let Sim handle jerk.
            constraints['accel_override'] = self.braking_accel
            
            # Allow physics to reach this target (don't clamp tight yet)
            # constraints['a_max'] = self.braking_accel # Removed to allow jerk transition
            # constraints['a_min'] = self.braking_accel # Removed
            
            # Check if stopped
            if v_current <= 0.1:
                self.state_name = "HOLDING"
                self.t_hold_start = t
                print(f"[STOP] Stopped at t={t:.2f}. Holding for {self.hold_duration}s...")
                
        if self.state_name == "HOLDING":
            # Force v=0
            constraints['v_limit'] = 0.0
            constraints['accel_override'] = 0.0 # Force zero accel target
            # constraints['a_max'] = 0.0 # Let jerk limiter handle the stop
            # constraints['a_min'] = 0.0
            
            # Check release
            if t >= self.t_hold_start + self.hold_duration:
                self.state_name = "RELEASED"
                print(f"[STOP] Released at t={t:.2f}. Resuming control.")
                
        # If RELEASED, do nothing (constraints passed through)
        
        return constraints

class SpeedLimitDisturbance(Disturbance):
    """Imposes a strict speed limit for a certain duration."""
    def __init__(self, t_start, limit, duration):
        super().__init__(t_start)
        self.limit = limit
        self.duration = duration
    
    def apply(self, t, state, constraints):
        if t >= self.t_start and t < self.t_start + self.duration:
            constraints['v_limit'] = min(constraints.get('v_limit', float('inf')), self.limit)
        return constraints

class TractionLossDisturbance(Disturbance):
    """Reduces maximum acceleration capability."""
    def __init__(self, t_start, factor, duration):
        super().__init__(t_start)
        self.factor = factor
        self.duration = duration
        
    def apply(self, t, state, constraints):
        if t >= self.t_start and t < self.t_start + self.duration:
            constraints['a_max'] *= self.factor
        return constraints
