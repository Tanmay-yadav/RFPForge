import subprocess
import time
import sys
import webbrowser

# Commands
BACKEND_CMD = [sys.executable, "-m", "uvicorn", "app.main:app"]
FRONTEND_CMD = [sys.executable, "-m", "streamlit", "run", "frontend/app.py"]


def start_backend():
    print("Starting FastAPI backend...")
    return subprocess.Popen(BACKEND_CMD)


def start_frontend():
    print("Starting Streamlit frontend...")
    return subprocess.Popen(FRONTEND_CMD)


if __name__ == "__main__":
    backend_process = None
    frontend_process = None

    try:
        backend_process = start_backend()

        # Give backend time to start
        print("Waiting for backend to initialize...")
        time.sleep(5)

        frontend_process = start_frontend()

        # Open browser automatically
        time.sleep(3)
        webbrowser.open("http://localhost:8501")

        print("\nSystem Running:")
        print("Backend:  http://127.0.0.1:8000")
        print("Frontend: http://localhost:8501\n")

        # Keep processes alive
        backend_process.wait()
        frontend_process.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")

        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()

        sys.exit(0)
