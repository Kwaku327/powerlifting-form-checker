"""
AI-Powered Powerlifting App
--------------------------
A computer vision application for assessing powerlifting attempts according to IPF rules.
"""

from .models import LiftResult, JointAngles, FrameData, LiftData
from .pose_estimation import PoseEstimator, process_video
from .rule_checker import RuleChecker, extract_joint_angles
from .utils import overlay_skeleton_and_annotations, get_key_frames, create_report

__version__ = "0.1.0" 
