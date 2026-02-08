import subprocess
import os
import sys

def start_system():
    print("Initializing AI Exam Monitoring System...")
    
    # 1. Start FastAPI Backend
    print("Starting Backend API...")
    backend_path = os.path.join("src", "api", "main.py")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    try:
        subprocess.run([sys.executable, backend_path], check=True, env=env)
    except KeyboardInterrupt:
        print("\nSystem shut down.")
    except Exception as e:
        print(f"Error starting system: {e}")

if __name__ == "__main__":
    start_system()
