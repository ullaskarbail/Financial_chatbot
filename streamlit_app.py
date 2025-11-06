"""
Streamlit Frontend for Personal Finance Chatbot
Main user interface application
"""

import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Personal Finance Chatbot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple, clean styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_api_status():
    """Get detailed API and AI service status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def call_api(endpoint: str, data: Dict[str, Any], timeout: int = 90) -> Dict[str, Any]:
    """Make API call to backend with configurable timeout"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/{endpoint}",
            json=data,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. The AI models might be loading for the first time (this can take up to 2 minutes). Please try again in a moment.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to the backend server. Please make sure it's running on http://localhost:8000")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def display_chat_interface():
    """Display the main chat interface"""
    st.title("💬 Financial Advisor Chat")
    
    # Get AI service status for display
    api_status = get_api_status()
    if api_status and "ai_services" in api_status:
        ai_services = api_status["ai_services"]
        
        # Display AI model status
        st.info(
            f"🤖 **AI Status**: "
            f"{'✅ Models Ready' if ai_services.get('models_loaded') else '⏳ Models Loading' if ai_services.get('models_loading') else '🔄 Will Load on First Use'} | "
            f"{'🔴 Fallback Mode' if ai_services.get('fallback_mode') else '🟢 AI Mode'}"
        )
        
        # Warning for first-time users
        if not ai_services.get('models_loaded') and not ai_services.get('models_loading'):
            st.warning(
                "⏰ **First Request Notice**: The AI models will load on your first question, "
                "which may take 1-2 minutes. Please be patient!"
            )
    
    # Initialize chat history and response
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_response" not in st.session_state:
        st.session_state.current_response = None
    
    # Simple form for chat
    with st.form("chat_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Ask your financial question:",
                placeholder="e.g., How can I save money while paying off student loans?",
                height=100
            )
        
        with col2:
            persona = st.selectbox(
                "Profile:",
                ["general", "student", "professional"],
                format_func=lambda x: x.title()
            )
            submit_button = st.form_submit_button("Send", type="primary")
    
    # Process user input
    if submit_button and user_input.strip():
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Call API
        with st.spinner("Getting response..."):
            response = call_api("chat", {
                "message": user_input,
                "persona": persona
            })
        
        if response:
            st.session_state.chat_history.append({
                "role": "bot",
                "content": response["response"],
                "nlu_analysis": response.get("nlu_analysis"),
                "timestamp": datetime.now()
            })
            st.session_state.current_response = response["response"]
            st.success("Response received!")
            st.rerun()
        else:
            st.error("Failed to get response. Please try again.")
    
    # Display current response in a dedicated window
    if st.session_state.current_response:
        st.subheader("💡 Current Response")
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border: 1px solid #e0e0e0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            {st.session_state.current_response.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
        
        # Clear current response button
        if st.button("Clear Response"):
            st.session_state.current_response = None
            st.rerun()
    else:
        st.info("Ask a financial question above to get personalized advice!")

def display_budget_summary():
    """Display budget summary analysis interface"""
    st.title("📊 Budget Summary Analysis")
    
    with st.form("budget_form"):
        st.subheader("Enter Your Financial Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)
            savings_goal = st.number_input("Monthly Savings Goal ($)", min_value=0.0, value=500.0, step=50.0)
            persona = st.selectbox(
                "Profile Type:",
                ["general", "student", "professional"],
                format_func=lambda x: x.title()
            )
        
        with col2:
            st.subheader("Monthly Expenses")
            rent = st.number_input("Rent/Mortgage", min_value=0.0, value=1500.0, step=50.0)
            utilities = st.number_input("Utilities", min_value=0.0, value=200.0, step=10.0)
            groceries = st.number_input("Groceries", min_value=0.0, value=400.0, step=25.0)
            transportation = st.number_input("Transportation", min_value=0.0, value=300.0, step=25.0)
            entertainment = st.number_input("Entertainment", min_value=0.0, value=200.0, step=25.0)
            other = st.number_input("Other Expenses", min_value=0.0, value=100.0, step=25.0)
        
        submit_button = st.form_submit_button("Analyze Budget", type="primary")
        
        if submit_button:
            expenses = {
                "Rent/Mortgage": rent,
                "Utilities": utilities,
                "Groceries": groceries,
                "Transportation": transportation,
                "Entertainment": entertainment,
                "Other": other
            }
            
            # Call API
            with st.spinner("Analyzing your budget..."):
                response = call_api("budget-summary", {
                    "income": income,
                    "expenses": expenses,
                    "savings_goal": savings_goal,
                    "persona": persona
                })
            
            if response:
                st.subheader("📈 Budget Analysis Results")
                
                # Financial metrics
                metrics = response["financial_metrics"]
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Income", f"${metrics['total_income']:,.2f}")
                with col2:
                    st.metric("Total Expenses", f"${metrics['total_expenses']:,.2f}")
                with col3:
                    st.metric("Disposable Income", f"${metrics['disposable_income']:,.2f}")
                with col4:
                    st.metric("Savings Rate", f"{metrics['savings_rate']:.1f}%")
                
                # Expense breakdown chart
                st.subheader("💰 Expense Breakdown")
                expense_data = []
                for category, amount in expenses.items():
                    if amount > 0:
                        expense_data.append({
                            "Category": category,
                            "Amount": amount,
                            "Percentage": (amount / metrics['total_expenses']) * 100
                        })
                
                if expense_data:
                    df = pd.DataFrame(expense_data)
                    fig = px.pie(df, values='Amount', names='Category', 
                               title="Monthly Expenses Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # AI Summary
                st.subheader("🤖 AI Analysis")
                st.write(response["summary"])
                
                # Recommendations
                if response["recommendations"]:
                    st.subheader("💡 Key Recommendations")
                    for i, rec in enumerate(response["recommendations"], 1):
                        st.write(f"{i}. {rec}")

def display_spending_insights():
    """Display spending insights analysis interface"""
    st.title("🔍 Spending Insights")
    
    with st.form("spending_form"):
        st.subheader("Enter Your Spending Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0, key="insights_income")
            persona = st.selectbox(
                "Profile Type:",
                ["general", "student", "professional"],
                format_func=lambda x: x.title(),
                key="insights_persona"
            )
        
        with col2:
            st.subheader("Financial Goals")
            goals = st.text_area(
                "What are your financial goals? (one per line)",
                placeholder="e.g., Save for emergency fund\nBuy a house\nPay off student loans",
                height=100
            )
        
        st.subheader("Monthly Expenses")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rent = st.number_input("Rent/Mortgage", min_value=0.0, value=1500.0, step=50.0, key="insights_rent")
            utilities = st.number_input("Utilities", min_value=0.0, value=200.0, step=10.0, key="insights_utilities")
            groceries = st.number_input("Groceries", min_value=0.0, value=400.0, step=25.0, key="insights_groceries")
        
        with col2:
            transportation = st.number_input("Transportation", min_value=0.0, value=300.0, step=25.0, key="insights_transport")
            entertainment = st.number_input("Entertainment", min_value=0.0, value=200.0, step=25.0, key="insights_entertainment")
            dining = st.number_input("Dining Out", min_value=0.0, value=150.0, step=25.0, key="insights_dining")
        
        with col3:
            shopping = st.number_input("Shopping", min_value=0.0, value=100.0, step=25.0, key="insights_shopping")
            insurance = st.number_input("Insurance", min_value=0.0, value=150.0, step=25.0, key="insights_insurance")
            other = st.number_input("Other", min_value=0.0, value=100.0, step=25.0, key="insights_other")
        
        submit_button = st.form_submit_button("Analyze Spending", type="primary")
        
        if submit_button:
            expenses = {
                "Rent/Mortgage": rent,
                "Utilities": utilities,
                "Groceries": groceries,
                "Transportation": transportation,
                "Entertainment": entertainment,
                "Dining Out": dining,
                "Shopping": shopping,
                "Insurance": insurance,
                "Other": other
            }
            
            goals_list = [goal.strip() for goal in goals.split('\n') if goal.strip()] if goals else []
            
            # Call API
            with st.spinner("Analyzing your spending patterns..."):
                response = call_api("spending-insights", {
                    "income": income,
                    "expenses": expenses,
                    "goals": goals_list,
                    "persona": persona
                })
            
            if response:
                st.subheader("📊 Spending Analysis Results")
                
                # Spending analysis metrics
                analysis = response["spending_analysis"]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Expenses", f"${analysis['total_expenses']:,.2f}")
                with col2:
                    st.metric("Essential Expenses", f"${analysis['essential_expenses']:,.2f}")
                with col3:
                    st.metric("Discretionary Expenses", f"${analysis['discretionary_expenses']:,.2f}")
                
                # Spending by category chart
                st.subheader("📈 Spending by Category")
                category_data = []
                for category, data in analysis['spending_by_category'].items():
                    if data['amount'] > 0:
                        category_data.append({
                            "Category": category,
                            "Amount": data['amount'],
                            "Percentage": data['percentage']
                        })
                
                if category_data:
                    df = pd.DataFrame(category_data)
                    fig = px.bar(df, x='Category', y='Amount', 
                               title="Monthly Spending by Category",
                               color='Amount',
                               color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
                
                # AI Insights
                st.subheader("🤖 AI Insights")
                st.write(response["insights"])
                
                # Action items
                if response["action_items"]:
                    st.subheader("✅ Action Items")
                    for i, item in enumerate(response["action_items"], 1):
                        st.write(f"{i}. {item}")

def display_nlu_analysis():
    """Display NLU analysis interface"""
    st.title("🔤 Text Analysis")
    
    st.subheader("Analyze Text with Natural Language Understanding")
    
    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="e.g., I'm struggling to save money because my expenses are too high and I have student loans to pay off.",
        height=150
    )
    
    if st.button("Analyze Text", type="primary") and text_input.strip():
        with st.spinner("Analyzing text..."):
            response = call_api("nlu", {"text": text_input})
        
        if response:
            st.subheader("📊 Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Analysis")
                sentiment = response["sentiment"]
                sentiment_label = sentiment.get("label", "neutral").title()
                sentiment_score = sentiment.get("score", 0.5)
                
                st.write(f"**Sentiment:** {sentiment_label}")
                st.write(f"**Score:** {sentiment_score:.2f}")
                
                # Simple sentiment gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sentiment_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Sentiment: {sentiment_label}"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "lightgreen"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Keywords")
                keywords = response["keywords"]
                if keywords:
                    for kw in keywords:
                        st.write(f"• **{kw.get('text', '')}** (relevance: {kw.get('relevance', 0):.2f})")
                else:
                    st.write("No keywords detected")
            
            # Entities and categories
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Entities")
                entities = response["entities"]
                if entities:
                    for entity in entities:
                        st.write(f"• **{entity.get('text', '')}** ({entity.get('type', 'unknown')})")
                else:
                    st.write("No entities detected")
            
            with col2:
                st.subheader("Categories")
                categories = response["categories"]
                if categories:
                    for category in categories:
                        st.write(f"• **{category.get('label', '')}** ({category.get('score', 0):.2f})")
                else:
                    st.write("No categories detected")

def display_chat_history():
    """Display chat history in a modal-like interface"""
    if "chat_history" not in st.session_state or not st.session_state.chat_history:
        st.info("No chat history available yet. Start a conversation first!")
        return
    
    st.subheader("📋 Conversation History")
    
    # Clear history button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.current_response = None
            st.success("Chat history cleared!")
            st.rerun()
    
    # Display messages with improved styling
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                color: white;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                border-left: 4px solid #0984e3;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <strong>🙋 You:</strong> {message["content"]}
                <br><small style="opacity: 0.8;">📅 {message["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
                color: white;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                border-left: 4px solid #00a085;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <strong>🤖 Financial Advisor:</strong> {message["content"].replace(chr(10), '<br>')}
                <br><small style="opacity: 0.8;">📅 {message["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}</small>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Check API health
    if not check_api_health():
        st.error("⚠️ Backend API is not running. Please start the FastAPI server first.")
        st.info("To start the server, run: `uvicorn main:app --reload`")
        return
    
    # Initialize session state for chat history visibility
    if "show_chat_history" not in st.session_state:
        st.session_state.show_chat_history = False
    
    # Simple sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["Chat", "Budget Summary", "Spending Insights", "Text Analysis"]
    )
    
    # Chat History button in sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("📋 Chat History", type="secondary"):
        st.session_state.show_chat_history = not st.session_state.show_chat_history
    
    # Display selected page
    try:
        if st.session_state.show_chat_history:
            display_chat_history()
        elif page == "Chat":
            display_chat_interface()
        elif page == "Budget Summary":
            display_budget_summary()
        elif page == "Spending Insights":
            display_spending_insights()
        elif page == "Text Analysis":
            display_nlu_analysis()
    except Exception as e:
        st.error(f"❌ Error loading {page}: {str(e)}")
    
    # Simple sidebar info
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.write("""
    **Personal Finance Chatbot**
    
    💬 **Chat**: Ask financial questions
    📊 **Budget**: Get budget summaries  
    🔍 **Insights**: Analyze spending
    🔤 **Analysis**: Understand text
    
    Powered by IBM Watson AI.
    """)
    
    # API status
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Status")
    if check_api_health():
        st.sidebar.success("✅ Backend Connected")
    else:
        st.sidebar.error("❌ Backend Disconnected")

if __name__ == "__main__":
    main()
