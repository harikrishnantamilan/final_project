import cv2
import numpy as np

try:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class BehaviorAnalyzer:
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp_pose
            self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.mp_draw = mp_drawing
        else:
            print("MediaPipe Pose not available.")
        
    def analyze_pose(self, frame):
        """Extract pose landmarks and detect anomalies."""
        if not hasattr(self, "pose"):
            return {}
            
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        pose_data = {}
        if results.pose_landmarks:
            pose_data["landmarks"] = results.pose_landmarks
            # Logic for head turning (using nose and ears)
            nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_ear = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            # Simple heuristic for head rotation
            head_yaw = (left_ear.x + right_ear.x) / 2 - nose.x
            pose_data["head_yaw"] = head_yaw
            
            # Logic for leaning behavior
            left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            pose_data["lean_score"] = abs(left_shoulder.y - right_shoulder.y)
            
        return pose_data

    def estimate_gaze(self, frame, pose_data):
        """Estimate gaze direction based on eye landmarks (simplified)."""
        # In a full implementation, we would extract the eye region and use a pre-trained model
        # Here we use head orientation as a proxy for gaze direction
        if "head_yaw" in pose_data:
            yaw = pose_data["head_yaw"]
            if abs(yaw) > 0.05:
                return "Sideways"
            return "Center"
        return "Unknown"

    def draw_pose(self, frame, pose_data):
        """Draw pose landmarks for visualization."""
        if not hasattr(self, "mp_draw"):
            return frame
        if "landmarks" in pose_data:
            self.mp_draw.draw_landmarks(frame, pose_data["landmarks"], self.mp_pose.POSE_CONNECTIONS)
        return frame
