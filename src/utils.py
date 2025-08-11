
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple
from .pose_estimation import PoseEstimator
from .models import LiftResult

def overlay_skeleton_and_annotations(frame: np.ndarray, landmarks, annotation: str = "", position: Tuple[int, int] = (30, 30)) -> np.ndarray:
    """Draw pose landmarks and add text annotation on the frame."""
    annotated_frame = frame.copy()
    annotated_frame = PoseEstimator.draw_landmarks(annotated_frame, landmarks)
    
    # Add text annotation
    if annotation:
        cv2.putText(
            annotated_frame,
            annotation,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),  # Red color for annotations
            2
        )
    return annotated_frame

def save_frame(frame: np.ndarray, path: str) -> None:
    """Save a frame to disk."""
    cv2.imwrite(path, frame)

def get_key_frames(video_path: str, frame_indices: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Extract key frames from video based on frame indices."""
    frames = {}
    cap = cv2.VideoCapture(video_path)
    
    for name, idx in frame_indices.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames[name] = frame
    
    cap.release()
    return frames

def create_report(lift_result, key_frames: Dict[str, np.ndarray], output_path: str) -> None:
    """Create a visual report with key frames and annotations."""
    # Create a white background
    height, width = key_frames[list(key_frames.keys())[0]].shape[:2]
    report = np.ones((height * 2, width * 2, 3), dtype=np.uint8) * 255
    
    # Add frames
    for i, (name, frame) in enumerate(key_frames.items()):
        row = i // 2
        col = i % 2
        y1 = row * height
        y2 = (row + 1) * height
        x1 = col * width
        x2 = (col + 1) * width
        report[y1:y2, x1:x2] = frame
        
        # Add frame title
        cv2.putText(
            report,
            name.replace('_', ' ').title(),
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2
        )
    
    # Add lift result
    result_text = "GOOD LIFT" if lift_result.is_good_lift else "NO LIFT"
    cv2.putText(
        report,
        result_text,
        (10, report.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0) if lift_result.is_good_lift else (0, 0, 255),
        3
    )
    
    # Add infractions
    y_pos = report.shape[0] - 60
    for infraction in lift_result.infractions:
        cv2.putText(
            report,
            f"- {infraction}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        y_pos -= 30

    # Save report
    cv2.imwrite(output_path, report)


def smooth_path(values: List[float], window: int = 5) -> List[float]:
    """Apply a centered moving average filter to the list of values."""
    if not values:
        return values
    if window < 2:
        return values
    pad = window // 2
    padded = [values[0]] * pad + values + [values[-1]] * pad
    smoothed = [float(np.mean(padded[i:i + window])) for i in range(len(values))]
    return smoothed


def export_result_json(lift_result: LiftResult, path: str) -> None:
    """Export LiftResult information to a JSON file."""
    with open(path, "w") as f:
        json.dump(lift_result.dict(), f, indent=2)
