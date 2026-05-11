import sys
import os
import threading
import webbrowser
import time
import streamlit.web.cli as stcli

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

    sys.argv = [
        "streamlit", "run", app_path,
        "--server.headless=true",
        "--server.port=8501",
        "--server.fileWatcherType=none",
        "--global.developmentMode=false"  
    ]
    
    sys.exit(stcli.main())
    
    sys.exit(stcli.main())

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        import multiprocessing
        multiprocessing.freeze_support()
        
    main()