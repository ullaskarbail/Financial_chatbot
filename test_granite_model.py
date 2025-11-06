#!/usr/bin/env python3
"""
Test script for Granite-3.0-2B-Instruct model integration
"""

import time
from ai_service import AIService

def main():
    print("🧪 Testing Granite-3.0-2B-Instruct Model Integration")
    print("=" * 60)
    
    # Initialize AI service
    print("🔧 Initializing AI Service with Granite model...")
    ai_service = AIService()
    
    # Check initial status
    status = ai_service.get_service_status()
    print(f"📊 Initial Status: {status}")
    print()
    
    # Test questions specifically for Granite model
    test_questions = [
        {
            "prompt": "What do you advise a student with a student loan and expenditure on gym?",
            "persona": "student"
        },
        {
            "prompt": "How should I prioritize paying off debt vs investing for retirement?",
            "persona": "professional"
        },
        {
            "prompt": "I want to save $10,000 in one year. What's the best strategy?",
            "persona": "general"
        }
    ]
    
    for i, test in enumerate(test_questions, 1):
        print(f"🎯 Test {i}: {test['persona'].title()} Persona")
        print(f"❓ Question: {test['prompt']}")
        print("⏳ Generating response with Granite model...")
        
        start_time = time.time()
        response = ai_service.generate_response(test['prompt'], test['persona'])
        end_time = time.time()
        
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📝 Response ({len(response)} chars):")
        print("-" * 50)
        print(response[:300] + "..." if len(response) > 300 else response)
        print("-" * 50)
        
        # Check if it's using Granite model or fallback
        status = ai_service.get_service_status()
        if status.get('models_loaded') and not status.get('fallback_mode'):
            print("✅ Using Granite-3.0-2B-Instruct model")
        else:
            print("⚠️ Using fallback responses (Granite model not loaded)")
        
        print()
        
        # Get updated status after first request
        if i == 1:
            updated_status = ai_service.get_service_status()
            print(f"📊 Status after first request: {updated_status}")
            print()
    
    # Test NLU analysis with Granite model loaded
    print("🧠 Testing NLU Analysis")
    test_text = "I'm excited about investing but worried about my student debt"
    nlu_result = ai_service.analyze_nlu(test_text)
    print(f"📄 Text: '{test_text}'")
    print(f"😊 Sentiment: {nlu_result['sentiment']}")
    print(f"🔑 Keywords: {nlu_result['keywords']}")
    print(f"🏷️  Categories: {nlu_result['categories']}")
    
    print("\n🎉 Granite model testing complete!")
    
    # Final status check
    final_status = ai_service.get_service_status()
    print(f"\n📋 Final Status Summary:")
    print(f"   Model: {'Granite-3.0-2B-Instruct' if final_status.get('models_loaded') else 'Fallback'}")
    print(f"   NLU Ready: {final_status.get('sentiment_analyzer_ready', False)}")
    print(f"   Text Gen Ready: {final_status.get('text_generator_ready', False)}")

if __name__ == "__main__":
    main()
