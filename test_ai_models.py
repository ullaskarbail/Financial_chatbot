#!/usr/bin/env python3
"""
Test script to verify AI model functionality
"""

import os
import sys
from ai_service import AIService

def test_ai_service():
    """Test the AI service functionality"""
    print("=" * 50)
    print("🧪 Testing AI Service")
    print("=" * 50)
    
    # Initialize AI service
    ai_service = AIService()
    
    # Check service status
    status = ai_service.get_service_status()
    print(f"\n📊 Service Status:")
    print(f"  - Hugging Face Available: {status['huggingface_available']}")
    print(f"  - Models Loaded: {status['models_loaded']}")
    
    # Test NLU analysis
    print(f"\n🔍 Testing NLU Analysis...")
    test_text = "I'm struggling with my budget and need help saving money for an emergency fund"
    nlu_result = ai_service.analyze_nlu(test_text)
    
    print(f"  Input: '{test_text}'")
    print(f"  Sentiment: {nlu_result['sentiment']}")
    print(f"  Keywords: {[kw['text'] for kw in nlu_result['keywords']]}")
    print(f"  Entities: {[ent['text'] for ent in nlu_result['entities']]}")
    print(f"  Categories: {[cat['label'] for cat in nlu_result['categories']]}")
    
    # Test response generation
    print(f"\n💬 Testing Response Generation...")
    test_prompts = [
        "How can I save money on groceries?",
        "What's the best way to pay off student loans?", 
        "Should I invest in stocks or bonds?"
    ]
    
    personas = ["general", "student", "professional"]
    
    for i, test_prompt in enumerate(test_prompts):
        persona = personas[i % len(personas)]
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: '{test_prompt}' (Persona: {persona})")
        
        response = ai_service.generate_response(test_prompt, persona)
        print(f"Response Length: {len(response)} chars")
        print(f"First 100 chars: {response[:100]}...")
        
        # Check if response is from model or fallback
        if "**Personal Finance Advice**" in response or "**Savings and Investment Guidance**" in response:
            print("⚠️  Used fallback response")
        else:
            print("✅ Used model response")
    
    print(f"\n✅ AI Service test completed!")

if __name__ == "__main__":
    test_ai_service()
