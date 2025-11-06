#!/usr/bin/env python3
"""
Test script for Personal Finance Chatbot
Tests the main functionality without requiring IBM Watson credentials
"""

import requests
import json
import time
import sys
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

def test_api_health():
    """Test if the API is running"""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API health check failed: {e}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint"""
    print("\n💬 Testing chat endpoint...")
    
    test_data = {
        "message": "How can I save money while paying off student loans?",
        "persona": "student"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=test_data,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Chat endpoint working")
            print(f"📝 Response length: {len(result.get('response', ''))} characters")
            print(f"🎭 Persona: {result.get('persona', 'unknown')}")
            return True
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Chat endpoint error: {e}")
        return False

def test_budget_summary_endpoint():
    """Test the budget summary endpoint"""
    print("\n📊 Testing budget summary endpoint...")
    
    test_data = {
        "income": 5000.0,
        "expenses": {
            "Rent": 1500.0,
            "Groceries": 400.0,
            "Transportation": 300.0,
            "Entertainment": 200.0,
            "Utilities": 200.0
        },
        "savings_goal": 500.0,
        "persona": "professional"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/budget-summary",
            json=test_data,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Budget summary endpoint working")
            print(f"📝 Summary length: {len(result.get('summary', ''))} characters")
            print(f"📈 Metrics calculated: {len(result.get('financial_metrics', {}))} items")
            print(f"💡 Recommendations: {len(result.get('recommendations', []))} items")
            return True
        else:
            print(f"❌ Budget summary endpoint failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Budget summary endpoint error: {e}")
        return False

def test_spending_insights_endpoint():
    """Test the spending insights endpoint"""
    print("\n🔍 Testing spending insights endpoint...")
    
    test_data = {
        "income": 5000.0,
        "expenses": {
            "Rent": 1500.0,
            "Groceries": 400.0,
            "Transportation": 300.0,
            "Entertainment": 200.0,
            "Dining": 150.0,
            "Shopping": 100.0
        },
        "goals": ["emergency fund", "vacation", "pay off debt"],
        "persona": "general"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/spending-insights",
            json=test_data,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Spending insights endpoint working")
            print(f"📝 Insights length: {len(result.get('insights', ''))} characters")
            print(f"📊 Analysis items: {len(result.get('spending_analysis', {}))} items")
            print(f"✅ Action items: {len(result.get('action_items', []))} items")
            return True
        else:
            print(f"❌ Spending insights endpoint failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Spending insights endpoint error: {e}")
        return False

def test_nlu_endpoint():
    """Test the NLU analysis endpoint"""
    print("\n🔤 Testing NLU analysis endpoint...")
    
    test_data = {
        "text": "I'm struggling to save money because my expenses are too high and I have student loans to pay off."
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/nlu",
            json=test_data,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ NLU analysis endpoint working")
            print(f"😊 Sentiment: {result.get('sentiment', {}).get('label', 'unknown')}")
            print(f"🔑 Keywords: {len(result.get('keywords', []))} found")
            print(f"🏷️ Entities: {len(result.get('entities', []))} found")
            print(f"📂 Categories: {len(result.get('categories', []))} found")
            return True
        else:
            print(f"❌ NLU analysis endpoint failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ NLU analysis endpoint error: {e}")
        return False

def test_features_endpoint():
    """Test the features endpoint"""
    print("\n📋 Testing features endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/features", timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Features endpoint working")
            print(f"🚀 Available features: {len(result.get('features', []))}")
            print(f"👥 Available personas: {len(result.get('personas', []))}")
            return True
        else:
            print(f"❌ Features endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Features endpoint error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Personal Finance Chatbot - Test Suite")
    print("=" * 60)
    
    tests = [
        ("API Health", test_api_health),
        ("Chat Endpoint", test_chat_endpoint),
        ("Budget Summary", test_budget_summary_endpoint),
        ("Spending Insights", test_spending_insights_endpoint),
        ("NLU Analysis", test_nlu_endpoint),
        ("Features", test_features_endpoint)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the application setup.")
        return False

if __name__ == "__main__":
    # Wait a moment for the server to start if it's not running
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    success = run_all_tests()
    
    if success:
        print("\n🚀 Ready to use the Personal Finance Chatbot!")
        print("📱 Open http://localhost:8501 in your browser")
    else:
        print("\n🔧 Please check the application setup and try again")
        sys.exit(1)
