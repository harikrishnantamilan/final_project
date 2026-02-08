import time
from src.config import WEIGHTS, TEMPORAL_WINDOW_SIZE

class MalpracticeEngine:
    def __init__(self):
        # student_id -> list of (timestamp, score_event)
        self.event_buffers = {}
        
    def calculate_malpractice_score(self, student_id, detections, gaze_status, lean_score):
        """Calculate a weighted malpractice score based on current detections."""
        if student_id not in self.event_buffers:
            self.event_buffers[student_id] = []
            
        current_score = 0
        
        # 1. Object Detections
        for det in detections:
            if "phone" in det["label"]:
                current_score += WEIGHTS["phone"]
            elif "chit" in det["label"] or "paper" in det["label"]:
                current_score += WEIGHTS["chit"]
                
        # 2. Behavioral Detections
        if gaze_status == "Sideways":
            current_score += WEIGHTS["gaze_deviation"]
            
        if lean_score > 0.2: # Threshold for unnatural leaning
            current_score += WEIGHTS["pose_anomaly"]
            
        # Add to temporal buffer
        self.event_buffers[student_id].append({
            "timestamp": time.time(),
            "score": current_score
        })
        
        # Maintain window size
        if len(self.event_buffers[student_id]) > TEMPORAL_WINDOW_SIZE:
            self.event_buffers[student_id].pop(0)
            
        # Aggregate score (temporal weighted smoothing)
        total_score = sum(e["score"] for e in self.event_buffers[student_id])
        avg_score = total_score / len(self.event_buffers[student_id])
        
        return self._classify_risk(avg_score)

    def _classify_risk(self, score):
        """Classify student risk level."""
        if score > 0.7:
            return "Malpractice Confirmed", score
        elif score > 0.4:
            return "High Risk", score
        elif score > 0.1:
            return "Mildly Suspicious", score
        return "Normal", score
