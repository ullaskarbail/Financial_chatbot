#!/usr/bin/env python3
"""
Startup script for Personal Finance Chatbot
Runs both the FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import os
import signal
import webbrowser
from threading import Thread

def run_backend():
    """Run the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def run_frontend():
    """Run the Streamlit frontend server"""
    print("🎨 Starting Streamlit frontend server...")
    time.sleep(3)  # Wait a bit for backend to start
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", 
            "run", "streamlit_app.py", 
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def main():
    """Main function to run both servers"""
    print("🤖 Personal Finance Chatbot - Startup Script")
    print("=" * 50)
    
    # Check if required files exist
    required_files = ["main.py", "streamlit_app.py", "ai_service.py", ".env"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    print("✅ All required files found")
    print("📋 Starting servers...")
    print()
    print("Backend will be available at: http://localhost:8000")
    print("Frontend will be available at: http://localhost:8501")
    print()
    print("🔧 Press Ctrl+C to stop both servers")
    print("=" * 50)
    
    try:
        # Start backend in a separate thread
        backend_thread = Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # Start frontend (this will block)
        run_frontend()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
