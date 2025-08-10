import numpy as np
from typing import List, Dict, Tuple
from .models import LiftData, LiftResult, FrameData

def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Return angle ABC (in degrees) between points a, b, c."""
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def extract_joint_angles(landmarks: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
    """Calculate relevant joint angles for squat, bench, deadlift from 2D landmarks."""
    angles = {}
    
    # Right hip angle = angle between right shoulder - right hip - right knee
    if all(k in landmarks for k in ['right_hip', 'right_shoulder', 'right_knee']):
        angles['hip_angle'] = calculate_angle(
            landmarks['right_shoulder'],
            landmarks['right_hip'],
            landmarks['right_knee']
        )
    else:
        angles['hip_angle'] = None

    # Right knee angle: hip - knee - ankle
    if all(k in landmarks for k in ['right_knee', 'right_hip', 'right_ankle']):
        angles['knee_angle'] = calculate_angle(
            landmarks['right_hip'],
            landmarks['right_knee'],
            landmarks['right_ankle']
        )
    else:
        angles['knee_angle'] = None

    # Right shoulder angle: hip - shoulder - elbow
    if all(k in landmarks for k in ['right_shoulder', 'right_hip', 'right_elbow']):
        angles['shoulder_angle'] = calculate_angle(
            landmarks['right_hip'],
            landmarks['right_shoulder'],
            landmarks['right_elbow']
        )
    else:
        angles['shoulder_angle'] = None

    # Right elbow angle: shoulder - elbow - wrist
    if all(k in landmarks for k in ['right_elbow', 'right_shoulder', 'right_wrist']):
        angles['elbow_angle'] = calculate_angle(
            landmarks['right_shoulder'],
            landmarks['right_elbow'],
            landmarks['right_wrist']
        )
    else:
        angles['elbow_angle'] = None

    return angles

class RuleChecker:
    @staticmethod
    def check_squat(lift_data: LiftData) -> LiftResult:
        """
        Returns LiftResult for squat.
        Logic:
        1. Find frame with minimal hip angle (deepest squat).
        2. Check if hip-angle indicates hip crease below knee: typically hip-angle < 90 degrees.
        3. Find highest frame (lockout) and check knee_angle near 180 degrees.
        4. Check for double-bounce: ensure no substantial dip after first ascent.
        """
        result = LiftResult(lift_type="squat", is_good_lift=True)
        angles_seq = [frame.angles for frame in lift_data.frames]

        # Identify lowest (deepest) point: minimal knee_angle
        knee_angle_tuples = [
            (i, f.angles.knee_angle)
            for i, f in enumerate(lift_data.frames)
            if f.angles.knee_angle is not None
        ]
        if not knee_angle_tuples:
            result.is_good_lift = False
            result.infractions.append("Unable to detect knee landmarks.")
            return result

        min_frame_idx, min_knee_angle = min(knee_angle_tuples, key=lambda x: x[1])
        # IPF requirement: Top of hip joint below top of knee => knee_angle ~ < 90 degrees
        if min_knee_angle > 95:  # small buffer above 90
            result.is_good_lift = False
            result.infractions.append(f"Insufficient depth: Knee angle at lowest point = {min_knee_angle:.1f}°")
        result.frame_indices['lowest_point'] = min_frame_idx

        # Check lockout: look at last frame
        last_frame = lift_data.frames[-1]
        if last_frame.angles.knee_angle < 170:
            result.is_good_lift = False
            result.infractions.append(f"Knees not fully locked at top: Knee angle = {last_frame.angles.knee_angle:.1f}°")
        result.frame_indices['lockout_frame'] = len(lift_data.frames) - 1

        # Check double-bounce: ensure knee_angle sequence is monotonically increasing after min_frame
        dipping_detected = False
        for i in range(min_frame_idx + 1, len(lift_data.frames) - 1):
            a1 = lift_data.frames[i].angles.knee_angle
            a2 = lift_data.frames[i + 1].angles.knee_angle
            if a1 is None or a2 is None:
                continue
            if a2 < a1 - 2:  # allow small noise
                dipping_detected = True
                break
        if dipping_detected:
            result.is_good_lift = False
            result.infractions.append("Double-bounce detected during ascent.")

        return result

    @staticmethod
    def check_bench(lift_data: LiftData) -> LiftResult:
        """
        Logic for bench press:
        1. Detect bar pause on chest: approximate by minimal vertical wrist movement.
        2. Check full arm extension at top (elbow_angle > 170°).
        3. Ensure no downward dip in bar after press begins.
        """
        result = LiftResult(lift_type="bench", is_good_lift=True)
        
        # Get wrist y-coordinates and elbow angles
        wrist_ys = []
        elbow_angles = []
        for frame in lift_data.frames:
            if 'right_wrist' in frame.landmarks:
                wrist_ys.append(frame.landmarks['right_wrist'][1])
            else:
                wrist_ys.append(None)
            elbow_angles.append(frame.angles.elbow_angle)

        # Find bottom position (chest touch)
        valid_wrist_ys = [(i, y) for i, y in enumerate(wrist_ys) if y is not None]
        if not valid_wrist_ys:
            result.is_good_lift = False
            result.infractions.append("Unable to detect wrist landmarks.")
            return result

        min_wrist_idx, _ = min(valid_wrist_ys, key=lambda x: x[1])
        result.frame_indices['chest_touch_frame'] = min_wrist_idx

        # Check pause at chest
        pause_frames = lift_data.frames[min_wrist_idx:min_wrist_idx + 5]
        pause_wrist_ys = [f.landmarks.get('right_wrist', (0, 0, 0))[1] for f in pause_frames]
        if max(pause_wrist_ys) - min(pause_wrist_ys) > 0.01:  # threshold on normalized coords
            result.is_good_lift = False
            result.infractions.append("No clear bar pause on chest.")

        # Check lockout
        last_elbow = elbow_angles[-1]
        if last_elbow < 170:
            result.is_good_lift = False
            result.infractions.append(f"Elbows not fully extended at top: Elbow angle = {last_elbow:.1f}°")
        result.frame_indices['lockout_frame'] = len(lift_data.frames) - 1

        # Check no downward movement during press
        for i in range(min_wrist_idx + 1, len(wrist_ys) - 1):
            if wrist_ys[i] is not None and wrist_ys[i + 1] is not None:
                if wrist_ys[i + 1] > wrist_ys[i] + 0.005:  # if bar moves down (y increases)
                    result.is_good_lift = False
                    result.infractions.append("Downward movement detected during press.")
                    break

        return result

    @staticmethod
    def check_deadlift(lift_data: LiftData) -> LiftResult:
        """
        Logic for deadlift:
        1. Check start position: slightly bent knees.
        2. Check lockout: full hip & knee extension.
        3. Ensure no hitching: no plateau in hip angle mid-lift.
        4. Ensure no downward movement before lockout.
        """
        result = LiftResult(lift_type="deadlift", is_good_lift=True)
        
        hip_angles = []
        knee_angles = []
        for frame in lift_data.frames:
            if frame.angles.hip_angle is not None:
                hip_angles.append(frame.angles.hip_angle)
            if frame.angles.knee_angle is not None:
                knee_angles.append(frame.angles.knee_angle)

        if not hip_angles or not knee_angles:
            result.is_good_lift = False
            result.infractions.append("Unable to detect hip or knee landmarks.")
            return result

        # Check lockout
        last_hip = hip_angles[-1]
        last_knee = knee_angles[-1]
        if last_hip < 170 or last_knee < 170:
            result.is_good_lift = False
            result.infractions.append(f"Incomplete lockout: Hip = {last_hip:.1f}°, Knee = {last_knee:.1f}°")
        result.frame_indices['lockout_frame'] = len(lift_data.frames) - 1

        # Check for hitching (non-monotonic hip angle)
        for i in range(1, len(hip_angles)):
            if hip_angles[i] < hip_angles[i - 1] - 2:  # allow small noise
                result.is_good_lift = False
                result.infractions.append("Hitching detected (hip angle decreased during lift).")
                break

        # Check no downward movement before lockout
        for i in range(len(hip_angles) - 1):
            if hip_angles[i] > 170 and hip_angles[i + 1] < hip_angles[i] - 2:
                result.is_good_lift = False
                result.infractions.append("Downward movement detected before lockout.")
                break

        return result

    @staticmethod
    def assess_lift(lift_type: str, lift_data: LiftData) -> LiftResult:
        """Main entry point for lift assessment."""
        if lift_type.lower() == 'squat':
            return RuleChecker.check_squat(lift_data)
        elif lift_type.lower() == 'bench':
            return RuleChecker.check_bench(lift_data)
        elif lift_type.lower() == 'deadlift':
            return RuleChecker.check_deadlift(lift_data)
        else:
            raise ValueError(f"Unknown lift type: {lift_type}") 