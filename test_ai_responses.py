#!/usr/bin/env python3
"""
Quick test script to verify AI models are working and generating real responses
"""

import sys
import time
from ai_service import AIService

def main():
    print("🧪 Testing AI Response Generation")
    print("=" * 50)
    
    # Initialize AI service
    print("🔧 Initializing AI Service...")
    ai_service = AIService()
    
    # Check initial status
    status = ai_service.get_service_status()
    print(f"📊 Initial Status: {status}")
    print()
    
    # Test questions
    test_questions = [
        {
            "prompt": "I want to save $1000 in 6 months. What should I do?",
            "persona": "student"
        },
        {
            "prompt": "Should I invest in stocks or real estate first?",
            "persona": "professional"
        },
        {
            "prompt": "How do I create a budget when my income varies each month?",
            "persona": "general"
        }
    ]
    
    for i, test in enumerate(test_questions, 1):
        print(f"🎯 Test {i}: {test['persona'].title()} Persona")
        print(f"❓ Question: {test['prompt']}")
        print("⏳ Generating response...")
        
        start_time = time.time()
        response = ai_service.generate_response(test['prompt'], test['persona'])
        end_time = time.time()
        
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📝 Response ({len(response)} chars):")
        print("-" * 30)
        print(response[:200] + "..." if len(response) > 200 else response)
        print("-" * 30)
        
        # Check if it's a hardcoded response
        hardcoded_indicators = ["**Budget Summary Analysis**", "**Smart Grocery Savings**", "**Personal Finance Advice**"]
        is_hardcoded = any(indicator in response for indicator in hardcoded_indicators)
        
        if is_hardcoded:
            print("⚠️  This appears to be a HARDCODED response (fallback mode)")
        else:
            print("✅ This appears to be an AI-GENERATED response")
        
        print()
        
        # Get updated status after first request
        if i == 1:
            updated_status = ai_service.get_service_status()
            print(f"📊 Status after first request: {updated_status}")
            print()
    
    # Test NLU
    print("🧠 Testing NLU Analysis")
    test_text = "I'm worried about my debt and struggling to pay my bills"
    nlu_result = ai_service.analyze_nlu(test_text)
    print(f"📄 Text: '{test_text}'")
    print(f"😊 Sentiment: {nlu_result['sentiment']}")
    print(f"🔑 Keywords: {nlu_result['keywords']}")
    print(f"🏷️  Categories: {nlu_result['categories']}")
    
    print("\n🎉 Testing complete!")

if __name__ == "__main__":
    main()
