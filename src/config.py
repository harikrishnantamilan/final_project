import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data & Models
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FACE_DATABASE_DIR = os.path.join(DATA_DIR, "face_database")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8", "yolov8n.pt")  # Default weights

# Detection Settings
PROHIBITED_OBJECTS = ["cell phone", "paper", "chit"]
DETECTION_THRESHOLD = 0.5

# Malpractice Scoring Weights
WEIGHTS = {
    "phone": 1.0,
    "chit": 0.8,
    "gaze_deviation": 0.4,
    "pose_anomaly": 0.3,
    "natural_movement_threshold": 0.2
}

# Monitoring Parameters
FPS_TARGET = 10
MAX_STUDENTS_PER_CAMERA = 60
TEMPORAL_WINDOW_SIZE = 30  # Number of frames for smoothing

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
