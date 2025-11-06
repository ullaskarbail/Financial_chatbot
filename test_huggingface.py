"""
Test script for Hugging Face integration with WatsonX models
"""

import os
from dotenv import load_dotenv
from ai_service import AIService

# Load environment variables
load_dotenv()

def test_huggingface_integration():
    """Test the Hugging Face integration"""
    print("Testing Hugging Face integration with WatsonX models...")
    
    # Initialize AI service
    ai_service = AIService()
    
    # Check service status
    status = ai_service.get_service_status()
    print(f"Service Status: {status}")
    
    # Test NLU analysis
    test_text = "I'm struggling to save money because my expenses are too high and I have student loans to pay off."
    print(f"\nTesting NLU analysis with text: '{test_text}'")
    
    nlu_result = ai_service.analyze_nlu(test_text)
    print(f"NLU Analysis Result:")
    print(f"  Sentiment: {nlu_result.get('sentiment', {})}")
    print(f"  Keywords: {nlu_result.get('keywords', [])}")
    print(f"  Entities: {nlu_result.get('entities', [])}")
    print(f"  Categories: {nlu_result.get('categories', [])}")
    
    # Test response generation
    test_prompt = "How can I save money while paying off student loans?"
    print(f"\nTesting response generation with prompt: '{test_prompt}'")
    
    response = ai_service.generate_response(test_prompt, persona="student")
    print(f"Generated Response:")
    print(response)
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_huggingface_integration()
