import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class PoseEstimator:
    def __init__(self, static_image_mode: bool = False, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_landmarks(self, frame: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        """
        Process a single frame and return a dictionary of landmark coordinates.
        Returns:
            {
              'left_hip': (x, y, z),
              'right_hip': (x, y, z),
              ...
            }
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        landmarks = {}

        if results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                # map idx to name
                name = mp_pose.PoseLandmark(idx).name.lower()
                landmarks[name] = (lm.x, lm.y, lm.z)
        return landmarks

    @staticmethod
    def draw_landmarks(frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw pose landmarks on frame."""
        if landmarks:
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            )
        return frame

def process_video(video_path: str, desired_fps: int = 30) -> List[Dict[str, Tuple[float, float, float]]]:
    """
    Read video file, extract frames at desired_fps, and return a list of landmark dictionaries per frame.
    """
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or desired_fps
    frame_interval = int(round(original_fps / desired_fps))

    estimator = PoseEstimator()
    landmarks_sequence = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            landmarks = estimator.extract_landmarks(frame)
            landmarks_sequence.append(landmarks)
        frame_count += 1

    cap.release()
    return landmarks_sequence 