"""
FastAPI Backend for Personal Finance Chatbot
Main application file with API endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import json
import os
from dotenv import load_dotenv

# Import our modules
from ai_service import AIService
from utils import (
    build_basic_prompt, 
    build_budget_summary_prompt, 
    build_spending_insights_prompt,
    validate_financial_data,
    format_currency,
    calculate_percentage
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Personal Finance Chatbot API",
    description="AI-powered personal finance advisor with Hugging Face integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI service
ai_service = AIService()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    persona: Optional[str] = "general"

class BudgetSummaryRequest(BaseModel):
    income: float
    expenses: Dict[str, float]
    savings_goal: float
    persona: Optional[str] = "general"

class SpendingInsightsRequest(BaseModel):
    income: float
    expenses: Dict[str, float]
    goals: Optional[List[str]] = []
    persona: Optional[str] = "general"

class NLURequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str
    nlu_analysis: Optional[Dict[str, Any]] = None
    persona: str

class BudgetSummaryResponse(BaseModel):
    summary: str
    financial_metrics: Dict[str, Any]
    recommendations: List[str]

class SpendingInsightsResponse(BaseModel):
    insights: str
    spending_analysis: Dict[str, Any]
    action_items: List[str]

class NLUResponse(BaseModel):
    sentiment: Dict[str, Any]
    keywords: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    categories: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Personal Finance Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "ai_services": ai_service.get_service_status()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_services": ai_service.get_service_status()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for general financial questions
    
    Args:
        request: ChatRequest with message and optional persona
        
    Returns:
        ChatResponse with AI-generated response and NLU analysis
    """
    try:
        # Analyze the user's message with NLU
        nlu_analysis = ai_service.analyze_nlu(request.message)
        
        # Build appropriate prompt based on the message
        prompt = build_basic_prompt(request.message, request.persona)
        
        # Generate response with persona
        response = ai_service.generate_response(prompt, request.persona)
        
        return ChatResponse(
            response=response,
            nlu_analysis=nlu_analysis,
            persona=request.persona
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.post("/budget-summary", response_model=BudgetSummaryResponse)
async def budget_summary_endpoint(request: BudgetSummaryRequest):
    """
    Budget summary analysis endpoint
    
    Args:
        request: BudgetSummaryRequest with financial data
        
    Returns:
        BudgetSummaryResponse with analysis and recommendations
    """
    try:
        # Validate and clean the data
        validated_data = validate_financial_data({
            'income': request.income,
            'expenses': request.expenses,
            'savings_goal': request.savings_goal,
            'persona': request.persona
        })
        
        # Build budget summary prompt
        prompt = build_budget_summary_prompt(
            validated_data['income'],
            validated_data['expenses'],
            validated_data['savings_goal'],
            validated_data['persona']
        )
        
        # Generate budget summary
        summary = ai_service.generate_response(prompt)
        
        # Calculate financial metrics
        total_expenses = sum(validated_data['expenses'].values())
        disposable_income = validated_data['income'] - total_expenses
        savings_rate = calculate_percentage(validated_data['savings_goal'], validated_data['income'])
        expense_rate = calculate_percentage(total_expenses, validated_data['income'])
        
        # Find top spending categories
        sorted_expenses = sorted(validated_data['expenses'].items(), key=lambda x: x[1], reverse=True)
        top_categories = sorted_expenses[:3]
        
        financial_metrics = {
            'total_income': validated_data['income'],
            'total_expenses': total_expenses,
            'disposable_income': disposable_income,
            'savings_goal': validated_data['savings_goal'],
            'savings_rate': savings_rate,
            'expense_rate': expense_rate,
            'top_spending_categories': [
                {'category': cat, 'amount': amt, 'percentage': calculate_percentage(amt, total_expenses)}
                for cat, amt in top_categories
            ]
        }
        
        # Extract recommendations from summary
        recommendations = []
        if 'reduce' in summary.lower():
            recommendations.append("Consider reducing discretionary expenses")
        if 'emergency' in summary.lower():
            recommendations.append("Build an emergency fund")
        if 'automate' in summary.lower():
            recommendations.append("Set up automatic savings transfers")
        if 'track' in summary.lower():
            recommendations.append("Track your spending regularly")
        if 'review' in summary.lower():
            recommendations.append("Review and adjust your budget monthly")
        
        return BudgetSummaryResponse(
            summary=summary,
            financial_metrics=financial_metrics,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing budget summary: {str(e)}")

@app.post("/spending-insights", response_model=SpendingInsightsResponse)
async def spending_insights_endpoint(request: SpendingInsightsRequest):
    """
    Spending insights analysis endpoint
    
    Args:
        request: SpendingInsightsRequest with spending data and goals
        
    Returns:
        SpendingInsightsResponse with insights and action items
    """
    try:
        # Prepare data for analysis
        monthly_data = {
            'income': request.income,
            'expenses': request.expenses,
            'goals': request.goals,
            'persona': request.persona
        }
        
        # Build spending insights prompt
        prompt = build_spending_insights_prompt(monthly_data)
        
        # Generate insights
        insights = ai_service.generate_response(prompt)
        
        # Calculate spending analysis
        total_expenses = sum(request.expenses.values())
        spending_by_category = {
            category: {
                'amount': amount,
                'percentage': calculate_percentage(amount, total_expenses)
            }
            for category, amount in request.expenses.items()
        }
        
        # Categorize expenses
        essential_categories = ['rent', 'mortgage', 'utilities', 'groceries', 'insurance']
        discretionary_categories = ['entertainment', 'dining', 'shopping', 'travel']
        
        essential_expenses = sum(
            amount for category, amount in request.expenses.items()
            if any(essential in category.lower() for essential in essential_categories)
        )
        discretionary_expenses = sum(
            amount for category, amount in request.expenses.items()
            if any(discretionary in category.lower() for discretionary in discretionary_categories)
        )
        
        spending_analysis = {
            'total_expenses': total_expenses,
            'essential_expenses': essential_expenses,
            'discretionary_expenses': discretionary_expenses,
            'spending_by_category': spending_by_category,
            'essential_percentage': calculate_percentage(essential_expenses, total_expenses),
            'discretionary_percentage': calculate_percentage(discretionary_expenses, total_expenses)
        }
        
        # Generate action items based on analysis
        action_items = []
        if discretionary_expenses > essential_expenses * 0.5:
            action_items.append("Reduce discretionary spending")
        if essential_expenses > request.income * 0.7:
            action_items.append("Look for ways to reduce essential expenses")
        if total_expenses > request.income * 0.9:
            action_items.append("Create a strict budget to avoid overspending")
        if request.goals:
            action_items.append("Prioritize your financial goals")
        action_items.append("Track spending patterns for the next month")
        
        return SpendingInsightsResponse(
            insights=insights,
            spending_analysis=spending_analysis,
            action_items=action_items
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing spending insights: {str(e)}")

@app.post("/nlu", response_model=NLUResponse)
async def nlu_endpoint(request: NLURequest):
    """
    Natural Language Understanding endpoint
    
    Args:
        request: NLURequest with text to analyze
        
    Returns:
        NLUResponse with sentiment, keywords, entities, and categories
    """
    try:
        # Analyze text with NLU
        analysis = ai_service.analyze_nlu(request.text)
        
        return NLUResponse(
            sentiment=analysis.get('sentiment', {}),
            keywords=analysis.get('keywords', []),
            entities=analysis.get('entities', []),
            categories=analysis.get('categories', [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing NLU request: {str(e)}")

@app.get("/features")
async def get_features():
    """Get available features and their descriptions"""
    return {
        "features": [
            {
                "name": "General Chat",
                "endpoint": "/chat",
                "description": "Ask general financial questions and get personalized advice",
                "input": "message (string), persona (optional)"
            },
            {
                "name": "Budget Summary",
                "endpoint": "/budget-summary", 
                "description": "Get comprehensive budget analysis and recommendations",
                "input": "income, expenses, savings_goal, persona (optional)"
            },
            {
                "name": "Spending Insights",
                "endpoint": "/spending-insights",
                "description": "Analyze spending patterns and get behavioral insights",
                "input": "income, expenses, goals (optional), persona (optional)"
            },
            {
                "name": "NLU Analysis",
                "endpoint": "/nlu",
                "description": "Analyze text for sentiment, keywords, and entities",
                "input": "text"
            }
        ],
        "personas": [
            "student",
            "professional", 
            "general"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", 8000)),
        reload=True
    )
