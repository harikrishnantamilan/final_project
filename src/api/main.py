from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import os
from src.core.face.face_manager import FaceManager
from src.core.detection.object_detector import ObjectDetector
from src.core.pose.behavior_analyzer import BehaviorAnalyzer
from src.core.malpractice_engine import MalpracticeEngine

app = FastAPI(title="AI Exam Monitoring System")

# Initialize Cores
face_mgr = FaceManager()
obj_det = ObjectDetector()
bh_analyzer = BehaviorAnalyzer()
engine = MalpracticeEngine()

@app.get("/")
async def root():
    return {"status": "Monitoring System Active"}

@app.post("/enroll")
async def enroll_student(student_id: str = Form(...), file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
        
    success = face_mgr.enroll_student(student_id, temp_path)
    os.remove(temp_path)
    
    return {"student_id": student_id, "status": "Success" if success else "Failed"}

@app.post("/analyze_frame")
async def analyze_frame(camera_id: str = Form(...), file: UploadFile = File(...)):
    # Read frame
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid image"}
        
    # 1. Face Identification
    try:
        face_locs, face_names = face_mgr.identify_face(frame)
    except Exception as e:
        print(f"Error in face identification: {e}")
        face_locs, face_names = [], []
    
    results = []
    # If no faces identified but we have a frame, analyze the frame anyway (general detection)
    if not face_names:
        try:
            detections = obj_det.detect_prohibited_items(frame)
            pose_data = bh_analyzer.analyze_pose(frame)
            gaze = bh_analyzer.estimate_gaze(frame, pose_data)
            
            results.append({
                "student_id": "Unknown",
                "detections": [d["label"] for d in detections],
                "gaze": gaze,
                "risk_level": "N/A"
            })
        except Exception as e:
            print(f"Error in general detection: {e}")
    else:
        for (top, right, bottom, left), name in zip(face_locs, face_names):
            try:
                # 2. Behavior & Detection
                detections = obj_det.detect_prohibited_items(frame) # Global detect
                pose_data = bh_analyzer.analyze_pose(frame)
                gaze = bh_analyzer.estimate_gaze(frame, pose_data)
                
                # 3. Scoring
                risk_level, score = engine.calculate_malpractice_score(
                    name, detections, gaze, pose_data.get("lean_score", 0)
                )
                
                results.append({
                    "student_id": name,
                    "risk_level": risk_level,
                    "score": f"{score:.2f}",
                    "gaze": gaze,
                    "detections": [d["label"] for d in detections]
                })
            except Exception as e:
                print(f"Error in student analysis ({name}): {e}")
        
    return {"camera_id": camera_id, "students": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
