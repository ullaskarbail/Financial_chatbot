#!/usr/bin/env python3
"""
Direct test by importing the AI service
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

from ai_service import AIService

def test_direct():
    """Test AI service directly"""
    print("🧪 Testing AI Service directly...")
    
    # Create AI service instance
    ai_service = AIService()
    
    # Check status
    status = ai_service.get_service_status()
    print(f"Service Status: {status}")
    
    # Test a simple prompt
    test_prompt = "How can I save money each month?"
    
    print(f"\n💬 Testing prompt: '{test_prompt}'")
    print("Generating response...")
    
    response = ai_service.generate_response(test_prompt, "general")
    
    print(f"\nResponse Length: {len(response)} chars")
    print(f"Response Preview: {response[:200]}...")
    
    # Check if it's hardcoded
    if "**Personal Finance Advice**" in response:
        print("⚠️ Using hardcoded fallback response")
        
        # Let's debug why
        print(f"\nDebugging:")
        print(f"HF Available: {ai_service.hf_available}")
        print(f"Model Loaded: {ai_service.model is not None}")
        print(f"Tokenizer Loaded: {ai_service.tokenizer is not None}")
        
    else:
        print("✅ Using model-generated response!")

if __name__ == "__main__":
    test_direct()
