from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from enum import Enum

from speed_profile_generator.webapp import config

class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"

class SimulationRequest(BaseModel):
    run_id: str
    traj_id: str
    scenario: str
    respect_limits: bool = False
    duration_policy: str = "keep"
    
    # UI State (passed to backend for result packaging)
    plan_visibility: str = "feasible"  # "feasible", "raw", "both"
    show_gt: bool = False
    focus_zoom: bool = False
    
    # Internal Config
    h5_path: str = config.DEFAULT_H5_PATH
    params: Dict[str, Any] = Field(default_factory=dict)
    downsample_points: int = 2000
    animation_points: int = 600

class SeriesData(BaseModel):
    # Executed (Time domain)
    exec_t_s: List[float]
    exec_x_m: List[float]
    exec_v_mps: List[float]
    exec_a_mps2: List[float]
    
    # Reference (Active plan mapped to executed position)
    ref_v_mps_time: Optional[List[float]] = None
    
    # Plan (Space domain)
    plan_x_m: List[float]
    plan_v_mps: List[float]
    feasible_plan_x_m: Optional[List[float]] = None
    feasible_plan_v_mps: Optional[List[float]] = None
    
    # Ground Truth (independent grid)
    gt_x_m: Optional[List[float]] = None
    gt_v_mps_pos: Optional[List[float]] = None
    gt_v_mps_time: Optional[List[float]] = None
    
    # Environment
    x_limit_m: List[float]
    v_limit_mps: List[float]

class EventData(BaseModel):
    type: str
    t_s: float
    x_m: float
    label: str
    color: str

class SimulatorMeta(BaseModel):
    dt_s: float
    t0_s: float
    tf_s: float
    x0_m: float
    xf_m: float
    a_min: float
    a_max: float
    run_id: str
    traj_id: str
    scenario: str
    
    # Narrative fields
    start_station_id: str = "A"
    end_station_id: str = "B"
    t_disturb: Optional[float] = None
    t_stop_reached: Optional[float] = None
    t_release: Optional[float] = None
    t_replan: Optional[float] = None
    t_terminal_on: Optional[float] = None
    t_terminal_off: Optional[float] = None
    t_end: Optional[float] = None

class SimulationResult(BaseModel):
    meta: SimulatorMeta
    series: SeriesData
    events: List[EventData]
    metrics: Dict[str, Any]
    logs: List[str]

class JobStatusResponse(BaseModel):
    job_id: str
    state: JobState
    progress: float = 0.0
    error: Optional[str] = None
    result: Optional[SimulationResult] = None

class TrajectoryInfo(BaseModel):
    id: str
    duration: float
    length: float

class RunInfo(BaseModel):
    id: str
    path: str
