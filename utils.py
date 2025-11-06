"""
Utility functions for the Personal Finance Chatbot
Handles prompt building, data processing, and response formatting
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime

def build_basic_prompt(user_input: str, persona: str = "general") -> str:
    """
    Build a basic prompt for general financial questions
    
    Args:
        user_input: The user's question or input
        persona: User persona (student, professional, general)
    
    Returns:
        Formatted prompt string
    """
    base_prompt = f"""You are a helpful personal finance advisor. 
    
User Question: {user_input}

Please provide clear, actionable financial advice. Focus on:
- Practical steps the user can take
- Specific recommendations when possible
- Educational explanations for financial concepts
- Encouraging and supportive tone

Response:"""
    
    if persona.lower() == "student":
        base_prompt += "\n\nNote: Provide advice suitable for students with limited income and experience."
    elif persona.lower() == "professional":
        base_prompt += "\n\nNote: Provide advice suitable for professionals with established careers."
    
    return base_prompt

def build_budget_summary_prompt(income: float, expenses: Dict[str, float], 
                               savings_goal: float, persona: str = "general") -> str:
    """
    Build a prompt for budget summary analysis
    
    Args:
        income: Monthly income
        expenses: Dictionary of expense categories and amounts
        savings_goal: Monthly savings goal
        persona: User persona
    
    Returns:
        Formatted prompt for budget analysis
    """
    total_expenses = sum(expenses.values())
    disposable_income = income - total_expenses
    savings_rate = (savings_goal / income) * 100 if income > 0 else 0
    
    expense_breakdown = "\n".join([f"- {category}: ${amount:.2f}" for category, amount in expenses.items()])
    
    prompt = f"""Analyze this budget and provide a comprehensive summary:

Monthly Income: ${income:.2f}
Monthly Expenses: ${total_expenses:.2f}
Disposable Income: ${disposable_income:.2f}
Savings Goal: ${savings_goal:.2f} ({savings_rate:.1f}% of income)

Expense Breakdown:
{expense_breakdown}

Please provide:
1. **Budget Summary**: Overall assessment of financial health
2. **Top Spending Categories**: Identify highest expenses
3. **Savings Analysis**: Can they meet their savings goal?
4. **Recommendations**: 3-5 specific ways to improve their budget
5. **Risk Assessment**: Any concerning patterns or red flags

Format your response in a clear, structured manner."""

    if persona.lower() == "student":
        prompt += "\n\nFocus on student-friendly advice and realistic expectations."
    elif persona.lower() == "professional":
        prompt += "\n\nProvide more sophisticated analysis suitable for professionals."
    
    return prompt

def build_spending_insights_prompt(monthly_data: Dict[str, Any]) -> str:
    """
    Build a prompt for spending behavior analysis
    
    Args:
        monthly_data: Dictionary containing spending data and goals
    
    Returns:
        Formatted prompt for spending analysis
    """
    income = monthly_data.get('income', 0)
    expenses = monthly_data.get('expenses', {})
    goals = monthly_data.get('goals', [])
    persona = monthly_data.get('persona', 'general')
    
    total_expenses = sum(expenses.values())
    expense_categories = "\n".join([f"- {cat}: ${amt:.2f}" for cat, amt in expenses.items()])
    goals_list = "\n".join([f"- {goal}" for goal in goals]) if goals else "No specific goals mentioned"
    
    prompt = f"""Analyze this spending behavior and provide detailed insights:

Monthly Income: ${income:.2f}
Total Monthly Expenses: ${total_expenses:.2f}
Spending Rate: {(total_expenses/income)*100:.1f}% of income

Expense Categories:
{expense_categories}

Financial Goals:
{goals_list}

Please provide:
1. **Spending Pattern Analysis**: Identify trends and patterns
2. **Goal Feasibility**: Can they achieve their stated goals?
3. **Optimization Opportunities**: Where can they cut costs?
4. **Behavioral Insights**: What does their spending reveal about habits?
5. **Action Plan**: Specific steps to improve financial situation
6. **Risk Factors**: Any concerning spending behaviors

Provide actionable, specific advice."""

    if persona.lower() == "student":
        prompt += "\n\nConsider student lifestyle and constraints."
    elif persona.lower() == "professional":
        prompt += "\n\nProvide professional-level financial analysis."
    
    return prompt

def format_currency(amount: float) -> str:
    """Format amount as currency string"""
    return f"${amount:,.2f}"

def calculate_percentage(part: float, whole: float) -> float:
    """Calculate percentage with safety check"""
    return (part / whole * 100) if whole > 0 else 0

def validate_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean financial data
    
    Args:
        data: Raw financial data
    
    Returns:
        Cleaned and validated data
    """
    cleaned_data = {}
    
    # Validate income
    income = data.get('income', 0)
    cleaned_data['income'] = float(income) if income and float(income) >= 0 else 0
    
    # Validate expenses
    expenses = data.get('expenses', {})
    cleaned_data['expenses'] = {}
    for category, amount in expenses.items():
        try:
            amount_float = float(amount)
            if amount_float >= 0:
                cleaned_data['expenses'][category] = amount_float
        except (ValueError, TypeError):
            continue
    
    # Validate savings goal
    savings_goal = data.get('savings_goal', 0)
    cleaned_data['savings_goal'] = float(savings_goal) if savings_goal and float(savings_goal) >= 0 else 0
    
    # Validate persona
    persona = data.get('persona', 'general')
    cleaned_data['persona'] = persona if persona in ['student', 'professional', 'general'] else 'general'
    
    return cleaned_data

def create_summary_card(title: str, content: str, color: str = "blue") -> str:
    """
    Create a styled summary card for Streamlit
    
    Args:
        title: Card title
        content: Card content
        color: Color theme
    
    Returns:
        HTML string for the card
    """
    colors = {
        "blue": "#1f77b4",
        "green": "#2ca02c", 
        "red": "#d62728",
        "orange": "#ff7f0e"
    }
    
    card_color = colors.get(color, colors["blue"])
    
    return f"""
    <div style="
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {card_color};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    ">
        <h3 style="color: {card_color}; margin-top: 0;">{title}</h3>
        <p style="margin-bottom: 0;">{content}</p>
    </div>
    """

def extract_key_insights(response: str) -> Dict[str, str]:
    """
    Extract key insights from AI response
    
    Args:
        response: AI-generated response
    
    Returns:
        Dictionary of extracted insights
    """
    insights = {
        'summary': '',
        'recommendations': '',
        'risks': '',
        'next_steps': ''
    }
    
    # Simple extraction based on common patterns
    lines = response.split('\n')
    current_section = 'summary'
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if any(keyword in line.lower() for keyword in ['recommendation', 'suggestion', 'advice']):
            current_section = 'recommendations'
        elif any(keyword in line.lower() for keyword in ['risk', 'warning', 'concern']):
            current_section = 'risks'
        elif any(keyword in line.lower() for keyword in ['next', 'action', 'step']):
            current_section = 'next_steps'
        elif line.startswith('**') and line.endswith('**'):
            # Section header, skip
            continue
        else:
            if insights[current_section]:
                insights[current_section] += ' ' + line
            else:
                insights[current_section] = line
    
    return insights
