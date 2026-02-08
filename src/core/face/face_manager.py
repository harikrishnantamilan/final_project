import cv2
import os
import pickle
import numpy as np

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    try:
        import mediapipe.python.solutions.face_detection as mp_face
        MP_FACE_AVAILABLE = True
    except ImportError:
        MP_FACE_AVAILABLE = False

from src.config import FACE_DATABASE_DIR

class FaceManager:
    def __init__(self, db_path=FACE_DATABASE_DIR):
        self.db_path = db_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.db_file = os.path.join(self.db_path, "face_encodings.pkl")
        
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            
        if not FACE_REC_AVAILABLE and MP_FACE_AVAILABLE:
            self.mp_face_detection = mp_face
            self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from the database file."""
        if os.path.exists(self.db_file):
            with open(self.db_file, "rb") as f:
                data = pickle.load(f)
                self.known_face_encodings = data["encodings"]
                self.known_face_names = data["names"]
            print(f"Loaded {len(self.known_face_names)} faces from database.")

    def save_known_faces(self):
        """Save known faces to the database file."""
        if FACE_REC_AVAILABLE:
            with open(self.db_file, "wb") as f:
                data = {"encodings": self.known_face_encodings, "names": self.known_face_names}
                pickle.dump(data, f)

    def enroll_student(self, student_id, image_path):
        """Enroll a student by generating face embeddings from an image."""
        if not FACE_REC_AVAILABLE:
            print("Face recognition library not available. Skipping embedding generation.")
            self.known_face_names.append(student_id)
            return True
            
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(student_id)
            self.save_known_faces()
            return True
        return False

    def identify_face(self, frame, tolerance=0.6):
        """Identify a face in a frame."""
        if FACE_REC_AVAILABLE:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                
                face_names.append(name)
            return face_locations, face_names
        elif MP_FACE_AVAILABLE:
            # Fallback to MediaPipe Face Detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            face_locations = []
            face_names = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    top = int(bbox.ymin * h)
                    left = int(bbox.xmin * w)
                    bottom = int((bbox.ymin + bbox.height) * h)
                    right = int((bbox.xmin + bbox.width) * w)
                    face_locations.append((top, right, bottom, left))
                    face_names.append("Student_Detected") # Placeholder
            return face_locations, face_names
        
        return [], []
