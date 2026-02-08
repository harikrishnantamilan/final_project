import cv2
import requests
import json
import time

def start_camera_client(camera_id="cam_local", api_url="http://localhost:8004/analyze_frame"):
    """
    Captures video from the local webcam, sends frames to the backend for analysis,
    and displays the results.
    """
    print(f"Connecting to camera... (Press 'q' to quit)")
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Encode frame to send to API
        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'camera_id': camera_id}

        try:
            # Send frame to backend
            response = requests.post(api_url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                students = result.get("students", [])

                # Draw analysis results on the frame
                for student in students:
                    # Parse data
                    student_id = student.get("student_id", "Unknown")
                    risk = student.get("risk_level", "Unknown")
                    detections = student.get("detections", [])
                    gaze = student.get("gaze", "Unknown")
                    
                    # Construct label text
                    label_text = f"ID: {student_id} | Risk: {risk} | Gaze: {gaze}"
                    if detections:
                        label_text += f" | Det: {', '.join(detections)}"
                    
                    # Draw on frame (simplified, since backend doesn't return bounding boxes for everything in the 'students' list structure yet,
                    # but let's assume valid face/detection logic runs).
                    # Actually, the backend response structure 'students' is a bit complex in main.py.
                    # It returns a list of results.
                    # The main.py does NOT helpfully return bounding boxes in the final `students` list for drawing here easily 
                    # unless we modify main.py to pass them back.
                    # However, we can just display the text overlay at the top or bottom for now.
                    
                    # Let's display the overall status on the top-left
                    cv2.putText(frame, label_text, (10, 30 + 30 * students.index(student)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if risk == "High" else (0, 255, 0), 2)
            else:
                print(f"Server Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("Connection Error: Is the backend running?")
            cv2.putText(frame, "Backend Offline", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error: {e}")

        # Show the frame
        cv2.imshow('AI Exam Monitoring Client', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera_client()
