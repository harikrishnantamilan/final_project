import sys
from ultralytics import YOLO
import cv2
from src.config import YOLO_MODEL_PATH, PROHIBITED_OBJECTS, DETECTION_THRESHOLD

class ObjectDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model = YOLO(model_path)
        
    def detect_prohibited_items(self, frame):
        """Detect prohibited objects in a frame."""
        results = self.model(frame, conf=DETECTION_THRESHOLD, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                # YOLO class IDs: cell phone is usually 67 in COCO dataset
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                
                if label in PROHIBITED_OBJECTS or label == "cell phone":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2)
                    })
        return detections

    def draw_detections(self, frame, detections):
        """Draw detections on the frame for visualization."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame

if __name__ == "__main__":
    import numpy as np
    import os
    
    # Set PYTHONPATH if running directly
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    print(f"Initializing ObjectDetector with model: {YOLO_MODEL_PATH}")
    try:
        detector = ObjectDetector()
        print("Model loaded successfully.")
        
        # Test with a blank frame
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Testing Detection", (100, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        detections = detector.detect_prohibited_items(blank_frame)
        print(f"Detections on test frame: {detections}")
        
    except Exception as e:
        print(f"Error during standalone test: {e}")
