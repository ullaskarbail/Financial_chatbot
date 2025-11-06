#!/usr/bin/env python3
"""
Direct test of API functionality
"""

import requests
import json

def test_api_chat():
    """Test the chat endpoint directly"""
    
    # Start the backend first
    import subprocess
    import time
    import sys
    
    print("🚀 Starting backend server...")
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "main:app", 
        "--host", "0.0.0.0", "--port", "8000", "--reload"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(10)
    
    try:
        # Test health endpoint
        print("📡 Testing health endpoint...")
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Health Status: {health_response.json()}")
        
        # Test chat endpoint
        print("💬 Testing chat endpoint...")
        chat_data = {
            "message": "How can I save money each month?",
            "persona": "general"
        }
        
        chat_response = requests.post(
            "http://localhost:8000/chat",
            json=chat_data,
            timeout=30
        )
        
        if chat_response.status_code == 200:
            result = chat_response.json()
            print(f"\n✅ Chat Response:")
            print(f"Length: {len(result['response'])} chars")
            print(f"First 200 chars: {result['response'][:200]}...")
            
            # Check if it's a hardcoded response
            if "**Personal Finance Advice**" in result['response']:
                print("⚠️  Still getting hardcoded response!")
            else:
                print("✅ Getting model-generated response!")
        else:
            print(f"❌ Chat request failed: {chat_response.status_code}")
            print(f"Error: {chat_response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
    finally:
        # Clean up
        print("🛑 Stopping backend server...")
        backend_process.terminate()

if __name__ == "__main__":
    test_api_chat()
