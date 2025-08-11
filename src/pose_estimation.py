import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import urllib.request
import os

class PoseEstimator:
    def __init__(self):
        # Download COCO model files if they don't exist
        self.model_folder = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_folder, exist_ok=True)
        
        self.prototxt_path = os.path.join(self.model_folder, 'pose_deploy_linevec.prototxt')
        self.weights_path = os.path.join(self.model_folder, 'pose_iter_440000.caffemodel')
        
        if not os.path.exists(self.prototxt_path):
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt',
                self.prototxt_path
            )
        
        if not os.path.exists(self.weights_path):
            urllib.request.urlretrieve(
                'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel',
                self.weights_path
            )
        
        # Initialize OpenCV's DNN module
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.weights_path)
        
        # COCO Output Format
        self.pose_pairs = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
                          (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
                          (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
        
        self.mapped_indices = {
            'nose': 0,
            'neck': 1,
            'right_shoulder': 2,
            'right_elbow': 3,
            'right_wrist': 4,
            'left_shoulder': 5,
            'left_elbow': 6,
            'left_wrist': 7,
            'right_hip': 8,
            'right_knee': 9,
            'right_ankle': 10,
            'left_hip': 11,
            'left_knee': 12,
            'left_ankle': 13,
            'right_eye': 14,
            'left_eye': 15,
            'right_ear': 16,
            'left_ear': 17
        }

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
        frame_height, frame_width = frame.shape[:2]
        
        # Prepare the frame
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        
        # Forward pass to get output
        output = self.net.forward()
        
        # Get Height and Width
        H = output.shape[2]
        W = output.shape[3]
        
        landmarks = {}
        
        # For each keypoint
        for key, idx in self.mapped_indices.items():
            # Get probability map
            prob_map = output[0, idx, :, :]
            
            # Find global maxima of the probmap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(prob_map)
            
            # Scale the point to fit on the original image
            x = (frame_width * point[0]) / W
            y = (frame_height * point[1]) / H
            
            # Add a keypoint only if its confidence is greater than threshold
            if prob > 0.1:
                # Normalize coordinates to [0,1] range
                landmarks[key] = (x / frame_width, y / frame_height, 0.0)
        
        return landmarks

    @staticmethod
    def detect_barbell_y(frame: np.ndarray, landmarks: Dict[str, Tuple[float, float, float]]) -> Optional[float]:
        """Detect horizontal barbell line near the wrists using Canny + Hough."""
        if 'left_wrist' not in landmarks or 'right_wrist' not in landmarks:
            return None

        h, w = frame.shape[:2]
        # Determine ROI around wrists
        lx, ly, _ = landmarks['left_wrist']
        rx, ry, _ = landmarks['right_wrist']
        cx = int(((lx + rx) / 2) * w)
        cy = int(((ly + ry) / 2) * h)
        roi_h = int(0.2 * h)
        roi_w = int(0.6 * w)
        y1 = max(cy - roi_h // 2, 0)
        y2 = min(cy + roi_h // 2, h)
        x1 = max(cx - roi_w // 2, 0)
        x2 = min(cx + roi_w // 2, w)
        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=60, maxLineGap=10)
        if lines is None:
            return None

        # Filter near-horizontal lines and take average y
        ys = []
        for x1l, y1l, x2l, y2l in lines[:, 0]:
            if abs(y2l - y1l) <= 5 and abs(x2l - x1l) > 30:
                ys.append((y1l + y2l) / 2)
        if not ys:
            return None
        barbell_y = (y1 + int(np.mean(ys))) / h
        return barbell_y

    @staticmethod
    def draw_landmarks(frame: np.ndarray, landmarks: Dict[str, Tuple[float, float, float]]) -> np.ndarray:
        """Draw pose landmarks on frame."""
        if not landmarks:
            return frame
            
        frame_height, frame_width = frame.shape[:2]
        img = frame.copy()
        
        # Draw points
        for name, (x, y, _) in landmarks.items():
            x_px = int(x * frame_width)
            y_px = int(y * frame_height)
            cv2.circle(img, (x_px, y_px), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(img, name, (x_px, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        
        # Draw skeleton
        for pair in [(landmarks.get('right_shoulder'), landmarks.get('right_elbow')),
                    (landmarks.get('right_elbow'), landmarks.get('right_wrist')),
                    (landmarks.get('right_hip'), landmarks.get('right_knee')),
                    (landmarks.get('right_knee'), landmarks.get('right_ankle')),
                    (landmarks.get('left_shoulder'), landmarks.get('left_elbow')),
                    (landmarks.get('left_elbow'), landmarks.get('left_wrist')),
                    (landmarks.get('left_hip'), landmarks.get('left_knee')),
                    (landmarks.get('left_knee'), landmarks.get('left_ankle')),
                    (landmarks.get('right_shoulder'), landmarks.get('right_hip')),
                    (landmarks.get('left_shoulder'), landmarks.get('left_hip')),
                    (landmarks.get('right_shoulder'), landmarks.get('left_shoulder')),
                    (landmarks.get('right_hip'), landmarks.get('left_hip'))]:
            if pair[0] and pair[1]:
                pt1 = (int(pair[0][0] * frame_width), int(pair[0][1] * frame_height))
                pt2 = (int(pair[1][0] * frame_width), int(pair[1][1] * frame_height))
                cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        
        return img

def process_video(video_path: str, desired_fps: int = 30) -> List[Dict[str, Optional[float]]]:
    """Return sequence of landmarks and barbell positions for each processed frame."""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or desired_fps
    frame_interval = int(round(original_fps / desired_fps))

    estimator = PoseEstimator()
    frame_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            landmarks = estimator.extract_landmarks(frame)
            barbell_y = PoseEstimator.detect_barbell_y(frame, landmarks)
            frame_data.append({"landmarks": landmarks, "barbell_y": barbell_y})
        frame_count += 1

    cap.release()
    return frame_data