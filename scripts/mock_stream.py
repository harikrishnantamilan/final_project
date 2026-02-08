import cv2
import numpy as np
import time
import requests
import os

def generate_mock_exam_stream(camera_id="cam_01", student_id="student_101"):
    """Simulate an exam session by sending mock frames to the API."""
    url = "http://localhost:8004/analyze_frame"
    
    # Create a blank frame representing a student at a desk
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, f"Student: {student_id}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Simulate various scenarios
    scenarios = [
        {"desc": "Normal behavior", "action": lambda f: f},
        {"desc": "Cell phone detection", "action": lambda f: cv2.rectangle(f, (200, 200), (300, 300), (0, 255, 0), 2)},
        {"desc": "Looking sideways", "action": lambda f: cv2.circle(f, (400, 100), 20, (255, 0, 0), -1)}
    ]
    
    print(f"Starting mock stream for {camera_id}...")
    
    for scenario in scenarios:
        print(f"Scenario: {scenario['desc']}")
        test_frame = scenario["action"](frame.copy())
        
        # Encode for transport
        _, img_encoded = cv2.imencode('.jpg', test_frame)
        files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'camera_id': camera_id}
        
        try:
            response = requests.post(url, files=files, data=data)
            print(f"API Response: {response.json()}")
        except Exception as e:
            print(f"API Error: {e}")
            
        time.sleep(2)

if __name__ == "__main__":
    # Ensure the student is enrolled first (mock enrollment)
    # This requires a real face image or a mock bypass in FaceManager
    generate_mock_exam_stream()
