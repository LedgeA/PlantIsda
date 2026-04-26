import subprocess
import sys
import os
import threading
import webbrowser
import time

def open_browser():
    time.sleep(4)
    webbrowser.open("http://localhost:8501")

def main():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(base_path, "app.py")

    threading.Thread(target=open_browser, daemon=True).start()

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.headless=true",
        "--server.port=8501",
        "--server.fileWatcherType=none"  # Important: prevents crashes in frozen env
    ])

if __name__ == "__main__":
    main()