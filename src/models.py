from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional

class JointAngles(BaseModel):
    hip_angle: float
    knee_angle: float
    shoulder_angle: Optional[float]
    elbow_angle: Optional[float]
    # Additional angles as needed per lift

class FrameData(BaseModel):
    landmarks: Dict[str, Tuple[float, float, float]]  # from PoseEstimator
    angles: JointAngles

class LiftData(BaseModel):
    frames: List[FrameData]
    # Additional aggregated features if desired (e.g., vertical bar path)

class LiftResult(BaseModel):
    lift_type: str  # squat, bench, or deadlift
    is_good_lift: bool
    infractions: List[str] = Field(default_factory=list)
    frame_indices: Dict[str, int] = Field(default_factory=dict)
    # e.g., {'lowest_point': 45, 'lockout': 90} 
