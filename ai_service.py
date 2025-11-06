"""
AI Service Module for Personal Finance Chatbot
Handles Hugging Face integration for NLU and response generation
"""

import os
import json
import requests
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()

class AIService:
    """Service class for AI interactions using Hugging Face models"""
    
    def __init__(self):
        """Initialize AI service with lazy loading to prevent server hanging"""
        # Hugging Face token
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        # Text generation and NLU model names (using lighter models for better performance)
        # Prefer provider-agnostic env var, fallback to old WATSONX_MODEL_NAME for compatibility
        self.generation_model_name = os.getenv('TEXT_GENERATION_MODEL_NAME') or os.getenv('WATSONX_MODEL_NAME', 'distilgpt2')
        self.nlu_model_name = os.getenv('NLU_MODEL_NAME', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
        self.ner_model_name = os.getenv('NER_MODEL_NAME', 'dslim/bert-base-NER')
        
        # Initialize models to None (lazy loading)
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        self.text_generator = None
        self.tokenizer = None
        self.model = None
        self.models_loading = False
        self.models_loaded = False
        
        # Check if Hugging Face token is available
        self.hf_available = bool(self.hf_token)
        
        if self.hf_available:
            try:
                # Login to Hugging Face (but don't load models yet)
                login(token=self.hf_token)
                print("Successfully logged in to Hugging Face Hub")
                print("Models will be loaded on first request to prevent server startup delays")
            except Exception as e:
                print(f"Error logging in to Hugging Face: {e}")
                self.hf_available = False
        else:
            print("Warning: Hugging Face token not found. Using fallback responses.")
    
    def _initialize_models(self):
        """Initialize Hugging Face models for different tasks"""
        try:
            # Import heavy libraries lazily to avoid DLL issues at startup
            from transformers import (
                pipeline,
                AutoTokenizer,
                AutoModelForCausalLM,
            )
            import torch

            # Initialize sentiment analysis model
            print("Loading sentiment analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=self.nlu_model_name,
                token=self.hf_token
            )
            
            # Initialize NER model for entity extraction
            print("Loading NER model...")
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                token=self.hf_token,
                aggregation_strategy="simple"
            )
            
            # Initialize text generation model
            print("Loading text generation model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.generation_model_name,
                token=self.hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.generation_model_name,
                token=self.hf_token,
                torch_dtype=torch.float32  # Use float32 for CPU compatibility
            )
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.hf_available = False
    
    def analyze_nlu(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using Hugging Face models for NLU tasks
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with NLU analysis results
        """
        # Initialize models on first request if not already loaded
        if self.hf_available and not self.models_loaded and not self.models_loading:
            self.models_loading = True
            try:
                self._initialize_models()
                self.models_loaded = True
            except Exception as e:
                print(f"Failed to initialize models: {e}")
                self.hf_available = False
            finally:
                self.models_loading = False
        
        if not self.hf_available or not self.sentiment_analyzer or not self.ner_pipeline:
            return self._fallback_nlu_analysis(text)
        
        try:
            # Sentiment analysis
            sentiment_result = self.sentiment_analyzer(text)
            sentiment = {
                'label': sentiment_result[0]['label'].lower(),
                'score': sentiment_result[0]['score']
            }
            
            # Named Entity Recognition
            entities_result = self.ner_pipeline(text)
            entities = []
            for entity in entities_result:
                entities.append({
                    'text': entity['word'],
                    'type': entity['entity_group'],
                    'score': entity['score']
                })
            
            # Keyword extraction using NER results
            keywords = []
            for entity in entities_result:
                keywords.append({
                    'text': entity['word'],
                    'relevance': entity['score']
                })
            
            # Add financial keywords if not found in NER
            financial_keywords = [
                'money', 'budget', 'savings', 'expenses', 'income', 'debt',
                'investment', 'tax', 'retirement', 'emergency fund', 'credit',
                'loan', 'mortgage', 'insurance', 'spending', 'cost', 'price'
            ]
            
            text_lower = text.lower()
            for keyword in financial_keywords:
                if keyword in text_lower and not any(kw['text'].lower() == keyword for kw in keywords):
                    keywords.append({'text': keyword, 'relevance': 0.8})
            
            # Categories (simplified based on content)
            categories = []
            if any(word in text_lower for word in ['budget', 'expenses', 'spending']):
                categories.append({'label': 'Budget Management', 'score': 0.9})
            if any(word in text_lower for word in ['savings', 'investment', 'retirement']):
                categories.append({'label': 'Savings & Investment', 'score': 0.8})
            if any(word in text_lower for word in ['debt', 'loan', 'credit']):
                categories.append({'label': 'Debt Management', 'score': 0.7})
            
            return {
                'sentiment': sentiment,
                'keywords': keywords[:5],
                'entities': entities[:5],
                'categories': categories
            }
            
        except Exception as e:
            print(f"NLU analysis error: {e}")
            return self._fallback_nlu_analysis(text)
    
    def _fallback_nlu_analysis(self, text: str) -> Dict[str, Any]:
        """
        Fallback NLU analysis when Hugging Face models are not available
        
        Args:
            text: Text to analyze
            
        Returns:
            Simulated NLU analysis results
        """
        # Simple keyword extraction
        keywords = []
        text_lower = text.lower()
        
        financial_keywords = [
            'money', 'budget', 'savings', 'expenses', 'income', 'debt',
            'investment', 'tax', 'retirement', 'emergency fund', 'credit',
            'loan', 'mortgage', 'insurance', 'spending', 'cost', 'price'
        ]
        
        for keyword in financial_keywords:
            if keyword in text_lower:
                keywords.append({'text': keyword, 'relevance': 0.8})
        
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'improve', 'better', 'saving', 'profit']
        negative_words = ['bad', 'worse', 'debt', 'loss', 'expensive', 'struggle', 'problem']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = {'label': 'positive', 'score': 0.7}
        elif negative_count > positive_count:
            sentiment = {'label': 'negative', 'score': 0.6}
        else:
            sentiment = {'label': 'neutral', 'score': 0.5}
        
        return {
            'sentiment': sentiment,
            'keywords': keywords[:5],
            'entities': [],
            'categories': []
        }
    
    def generate_response(self, prompt: str, persona: str = "general") -> str:
        """
        Generate response using Granite model with intelligent fallback
        
        Args:
            prompt: Input prompt for the AI model
            persona: User persona (general, student, professional)
            
        Returns:
            Generated response text
        """
        # Initialize models on first request if not already loaded
        if self.hf_available and not self.models_loaded and not self.models_loading:
            self.models_loading = True
            try:
                print("First request detected - initializing AI models...")
                self._initialize_models()
                self.models_loaded = True
                print("AI models loaded successfully!")
            except Exception as e:
                print(f"Failed to initialize models: {e}")
                print("Falling back to enhanced rule-based responses")
                self.hf_available = False
            finally:
                self.models_loading = False
        
        print(f"Generating response for persona: {persona}")
        
        # Try to use Granite model first (should be much better than GPT-2)
        if self.hf_available and self.models_loaded and self.tokenizer and self.model:
            try:
                return self._generate_with_granite_model(prompt, persona)
            except Exception as e:
                print(f"Error generating with Granite model: {e}")
                print("Falling back to enhanced rule-based response")
        
        # Fallback to enhanced rule-based responses
        return self._enhanced_response_generation(prompt, persona)
    
    def _generate_with_granite_model(self, prompt: str, persona: str = "general") -> str:
        """
        Generate response using Granite-3.0-2B-Instruct model with instruction-based prompting
        
        Args:
            prompt: Input prompt
            persona: User persona
            
        Returns:
            Generated response text
        """
        try:
            # Create instruction-based prompt for Granite model
            system_instruction = "You are a professional financial advisor with expertise in personal finance, budgeting, investments, and debt management. Provide clear, actionable, and practical financial advice."
            
            # Persona-specific context
            if persona == "student":
                context = "Focus on student-specific financial challenges like student loans, part-time income, and building financial habits on a limited budget."
            elif persona == "professional":
                context = "Focus on professional-level financial planning including retirement, investments, tax strategies, and wealth building."
            else:
                context = "Provide general financial advice suitable for various income levels and life situations."
            
            # Format as instruction prompt for Granite model
            full_prompt = f"""<|system|>
{system_instruction} {context}
<|user|>
{prompt}
<|assistant|>
"""
            
            # Tokenize the prompt
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Generate with parameters optimized for instruction-following models
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=200,      # Longer responses for better advice
                    min_new_tokens=50,       # Ensure substantial response
                    do_sample=True,
                    temperature=0.3,         # Lower temperature for more focused responses
                    top_p=0.9,              # Nucleus sampling
                    top_k=50,               # Limit vocabulary for coherence
                    repetition_penalty=1.1,  # Avoid repetition
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant response
            if "<|assistant|>" in full_response:
                response = full_response.split("<|assistant|>")[-1].strip()
            else:
                # Fallback extraction method
                response = full_response[len(full_prompt):].strip()
            
            # Clean and validate the response
            response = self._clean_granite_response(response)
            
            if self._is_valid_granite_response(response, prompt):
                return response
            else:
                print("Granite response quality check failed, using fallback")
                return self._enhanced_response_generation(prompt, persona)
                
        except Exception as e:
            print(f"Error in Granite model generation: {e}")
            return self._enhanced_response_generation(prompt, persona)
    
    def _clean_granite_response(self, response: str) -> str:
        """
        Clean and format the response from Granite model
        
        Args:
            response: Raw response from model
            
        Returns:
            Cleaned response text
        """
        # Remove any system tokens that might have leaked through
        response = response.replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", "")
        
        # Clean up whitespace and formatting
        response = response.strip()
        
        # Remove incomplete sentences at the end
        sentences = response.split('. ')
        if len(sentences) > 1 and not sentences[-1].endswith(('.', '!', '?')):
            sentences = sentences[:-1]
        response = '. '.join(sentences)
        
        # Ensure proper sentence ending
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
    
    def _is_valid_granite_response(self, response: str, original_prompt: str) -> bool:
        """
        Validate if the Granite model response is appropriate and high-quality
        
        Args:
            response: Generated response
            original_prompt: Original user question
            
        Returns:
            True if response is valid and high-quality
        """
        # Check minimum length
        if len(response.strip()) < 50:
            return False
        
        # Check for financial content relevance
        response_lower = response.lower()
        financial_terms = [
            'budget', 'save', 'saving', 'money', 'financial', 'income', 'expense',
            'investment', 'debt', 'loan', 'credit', 'fund', 'account', 'plan',
            'goal', 'strategy', 'recommend', 'consider', 'advice'
        ]
        
        has_financial_content = any(term in response_lower for term in financial_terms)
        
        # Check response coherence (not too repetitive)
        words = response_lower.split()
        unique_words = set(words)
        coherence_ratio = len(unique_words) / len(words) if words else 0
        
        # Response should be coherent (>60% unique words) and contain financial advice
        return has_financial_content and coherence_ratio > 0.6 and len(response) > 50
    
    def _enhanced_response_generation(self, prompt: str, persona: str = "general") -> str:
        """
        Enhanced rule-based response generation with persona awareness
        
        Args:
            prompt: Input prompt
            persona: User persona
            
        Returns:
            High-quality financial advice response
        """
        prompt_lower = prompt.lower()
        
        # Student-specific financial advice
        if 'student' in prompt_lower or persona == 'student':
            if any(word in prompt_lower for word in ['loan', 'debt', 'payment']):
                return self._get_student_loan_advice(prompt_lower)
            elif any(word in prompt_lower for word in ['save', 'saving', 'money']):
                return self._get_student_saving_advice(prompt_lower)
            elif any(word in prompt_lower for word in ['budget', 'expense']):
                return self._get_student_budget_advice(prompt_lower)
            elif any(word in prompt_lower for word in ['gym', 'fitness', 'membership']):
                return self._get_student_gym_advice(prompt_lower)
        
        # Professional-specific advice
        elif persona == 'professional':
            if any(word in prompt_lower for word in ['investment', 'invest', 'portfolio']):
                return self._get_professional_investment_advice(prompt_lower)
            elif any(word in prompt_lower for word in ['retirement', '401k', 'pension']):
                return self._get_professional_retirement_advice(prompt_lower)
            elif any(word in prompt_lower for word in ['tax', 'deduction']):
                return self._get_professional_tax_advice(prompt_lower)
        
        # General financial topics
        if 'budget' in prompt_lower and 'summary' in prompt_lower:
            return self._fallback_response_generation(prompt, persona)
        elif any(word in prompt_lower for word in ['save', 'saving', 'money']):
            return self._get_general_saving_advice(prompt_lower)
        elif any(word in prompt_lower for word in ['debt', 'loan', 'credit']):
            return self._get_debt_management_advice(prompt_lower)
        elif any(word in prompt_lower for word in ['investment', 'invest']):
            return self._get_investment_advice(prompt_lower)
        elif any(word in prompt_lower for word in ['budget', 'expense']):
            return self._get_budget_advice(prompt_lower)
        elif any(word in prompt_lower for word in ['emergency', 'fund']):
            return self._get_emergency_fund_advice(prompt_lower)
        
        # Default general advice
        return self._fallback_response_generation(prompt, persona)
    
    def _generate_with_huggingface(self, prompt: str, persona: str = "general") -> str:
        """
        Generate response using Hugging Face text generation model with quality validation
        
        Args:
            prompt: Input prompt
            persona: User persona
            
        Returns:
            Generated response text
        """
        try:
            # Create a more structured prompt for better responses
            financial_context = """
You are a professional financial advisor. Provide clear, actionable, and practical financial advice.
Always structure your response with specific steps, recommendations, and explanations.
Focus on being helpful, accurate, and encouraging.
"""
            
            persona_context = self._get_persona_context(persona)
            
            # Create a well-structured prompt
            full_prompt = f"{financial_context}\n{persona_context}\n\nClient Question: {prompt}\n\nFinancial Advisor Response:"
            
            # Tokenize with better parameters
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=400,  # Shorter input for better context
                padding=True
            )
            
            # Generate with improved parameters for better quality
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=100,   # Shorter responses for better quality
                    min_new_tokens=30,    # Ensure meaningful length
                    do_sample=True,
                    temperature=0.4,      # Lower temperature for more focused responses
                    top_p=0.8,           # More focused sampling
                    top_k=50,            # Limit vocabulary for coherence
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.1,  # Penalize repetition
                    length_penalty=1.0
                )
            
            # Decode the response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part
            if "Financial Advisor Response:" in full_response:
                response = full_response.split("Financial Advisor Response:")[-1].strip()
            elif "Financial Advisor:" in full_response:
                response = full_response.split("Financial Advisor:")[-1].strip()
            else:
                response = full_response[len(full_prompt):].strip()
            
            # Validate response quality
            if self._is_valid_financial_response(response, prompt):
                return self._format_model_response(response, persona)
            else:
                print("Generated response quality too low, using fallback")
                return self._fallback_response_generation(prompt, persona)
                
        except Exception as e:
            print(f"Error in Hugging Face generation: {e}")
            return self._fallback_response_generation(prompt, persona)
    
    def _is_valid_financial_response(self, response: str, original_prompt: str) -> bool:
        """
        Validate if the generated response is appropriate for financial advice
        
        Args:
            response: Generated response
            original_prompt: Original user question
            
        Returns:
            True if response is valid financial advice
        """
        response_lower = response.lower()
        prompt_lower = original_prompt.lower()
        
        # Check minimum length
        if len(response.strip()) < 30:
            return False
        
        # Check if response contains financial keywords or advice
        financial_indicators = [
            'budget', 'save', 'saving', 'money', 'income', 'expense', 'debt',
            'investment', 'financial', 'plan', 'goal', 'recommend', 'consider',
            'reduce', 'increase', 'strategy', 'fund', 'account', 'loan'
        ]
        
        has_financial_content = any(word in response_lower for word in financial_indicators)
        
        # Check if response is coherent (not too repetitive or fragmented)
        sentences = response.split('.')
        if len(sentences) < 2:  # Too short or not structured
            return False
        
        # Check for excessive repetition
        words = response_lower.split()
        if len(set(words)) < len(words) * 0.6:  # Too much repetition
            return False
        
        # Check if response seems relevant to the question
        question_keywords = [word for word in prompt_lower.split() if len(word) > 3]
        response_keywords = [word for word in response_lower.split() if len(word) > 3]
        
        # At least some keyword overlap or financial content
        keyword_overlap = any(qw in response_keywords for qw in question_keywords[:5])
        
        return has_financial_content and (keyword_overlap or len(response) > 100)
    
    def _get_persona_context(self, persona: str) -> str:
        """Get context based on user persona"""
        contexts = {
            "student": """You are a financial advisor specializing in helping students manage their finances. 
            Focus on budgeting, student loans, part-time work, and building good financial habits early. 
            Be encouraging and provide practical, actionable advice for students with limited income.""",
            
            "professional": """You are a financial advisor for working professionals. 
            Focus on retirement planning, investment strategies, tax optimization, and wealth building. 
            Provide sophisticated advice while considering career growth and long-term financial goals.""",
            
            "general": """You are a knowledgeable and friendly financial advisor. 
            Provide clear, practical advice on personal finance topics including budgeting, saving, 
            investing, debt management, and financial planning. Be encouraging and actionable in your responses."""
        }
        
        return contexts.get(persona, contexts["general"])
    
    def _format_model_response(self, response: str, persona: str = "general") -> str:
        """
        Format and enhance the model response based on persona and financial context
        
        Args:
            response: Raw model response
            persona: User persona
            
        Returns:
            Formatted response text
        """
        # Clean up the response
        response = response.strip()
        
        # Remove incomplete sentences at the end
        sentences = response.split('. ')
        if len(sentences) > 1 and not sentences[-1].endswith(('.', '!', '?')):
            sentences = sentences[:-1]
        response = '. '.join(sentences)
        
        # Add persona-specific prefix if appropriate
        if persona == "student":
            if not response.startswith(('As a student', 'For students', 'When you\'re a student')):
                response = f"As a student, {response.lower()}"
        elif persona == "professional":
            if not response.startswith(('As a professional', 'For professionals', 'In your career')):
                response = f"As a working professional, {response.lower()}"
        
        # Ensure the response ends properly
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
    
    def _fallback_response_generation(self, prompt: str, persona: str = "general") -> str:
        """
        Fallback response generation when WatsonX model is not available
        
        Args:
            prompt: Input prompt
            persona: User persona
            
        Returns:
            Generated response text
        """
        # Simple rule-based responses for common financial topics
        prompt_lower = prompt.lower()
        
        if 'budget' in prompt_lower and 'summary' in prompt_lower:
            return """**Budget Summary Analysis**

Based on your financial data, here's my assessment:

**Overall Financial Health**: Your budget shows a [positive/neutral/concerning] financial situation.

**Key Insights**:
- Your total expenses represent [X]% of your income
- You have [positive/negative] disposable income each month
- Your savings goal is [achievable/challenging] given current spending

**Top Spending Categories**:
1. [Category 1]: [X]% of total expenses
2. [Category 2]: [X]% of total expenses

**Recommendations**:
1. Consider reducing spending in [specific category]
2. Set up automatic transfers to savings
3. Review discretionary expenses monthly
4. Build an emergency fund if you haven't already

**Risk Assessment**: [Any concerning patterns or red flags]

Remember to regularly review and adjust your budget as your circumstances change."""

        elif 'grocer' in prompt_lower and 'saving' in prompt_lower:
            return """**Smart Grocery Savings Strategies**

Here are proven ways to reduce your grocery expenses:

**Planning & Preparation**:
1. **Meal Planning**: Plan your meals for the week before shopping
2. **Shopping List**: Make a detailed list and stick to it
3. **Budget Setting**: Set a weekly/monthly grocery budget
4. **Inventory Check**: Check what you already have at home

**Shopping Strategies**:
- **Store Comparison**: Compare prices across different stores
- **Generic Brands**: Buy store brands instead of name brands (save 20-30%)
- **Bulk Buying**: Purchase non-perishables in bulk when on sale
- **Seasonal Shopping**: Buy fruits and vegetables in season
- **Cash/Card**: Use cash to avoid overspending

**Money-Saving Tips**:
- **Coupons & Apps**: Use store apps, digital coupons, and cashback apps
- **Sales Timing**: Shop during weekly sales and clearance events
- **Unit Prices**: Compare cost per unit, not just package price
- **Avoid Shopping Hungry**: Eat before grocery shopping
- **Loyalty Programs**: Join store loyalty programs for discounts

**Smart Food Choices**:
- **Cook at Home**: Reduce dining out and takeaway orders
- **Batch Cooking**: Prepare large portions and freeze extras
- **Protein Alternatives**: Include affordable proteins like beans, eggs, chicken
- **Frozen/Canned**: Buy frozen vegetables and canned goods when fresh is expensive

**Expected Savings**: With these strategies, you can reduce grocery bills by 15-30%.

Remember: Small changes in grocery habits can lead to significant savings over time!"""
        
        elif 'house' in prompt_lower and ('saving' in prompt_lower or 'buy' in prompt_lower):
            return """**Saving for a House Purchase**

Buying a house worth 5 crore (₹50 million) is a significant investment. Here's a strategic approach:

**Down Payment Planning**:
- Traditional down payment: 10-20% = ₹5-10 million
- Consider starting with a smaller target if this is your first home
- Factor in additional costs: registration, stamp duty, legal fees (~2-3% extra)

**Savings Strategy for Large Home Purchase**:
1. **Set a realistic timeline**: For ₹5-10 million down payment, plan 5-10 years
2. **Monthly savings target**: ₹50,000-₹1,00,000 per month
3. **High-yield investments**: Consider equity mutual funds, ELSS, PPF
4. **Systematic Investment Plans (SIPs)**: Automate your savings

**Financial Preparation**:
- **Income requirement**: Your monthly income should be 3-4x the EMI
- **Credit score**: Maintain 750+ for better loan terms
- **Debt-to-income ratio**: Keep below 40% including the new home loan
- **Emergency fund**: Maintain 6 months of expenses separately

**Alternative Approaches**:
- Consider starting with a smaller home and upgrading later
- Look into pre-approved loans to understand your borrowing capacity
- Explore different locations for better value
- Consider ready-to-move vs under-construction properties

**Smart Tips**:
- Track real estate market trends in your target area
- Factor in maintenance costs (1-2% of property value annually)
- Consider tax benefits under Section 80C and 24(b)

Remember: A house is both a home and an investment. Plan carefully and don't overstretch your finances."""
        
        elif 'saving' in prompt_lower or 'investment' in prompt_lower:
            return """**Savings and Investment Guidance**

Here are some practical steps to improve your financial situation:

**Immediate Actions**:
1. **Emergency Fund**: Aim to save 3-6 months of expenses
2. **Automate Savings**: Set up automatic transfers from checking to savings
3. **Track Spending**: Use apps or spreadsheets to monitor expenses
4. **Reduce Expenses**: Look for areas to cut back on non-essential spending

**Investment Considerations**:
- Start with employer retirement plans (401k, 403b)
- Consider Roth IRA for tax-free growth
- Diversify investments across different asset classes
- Start early to benefit from compound interest

**Smart Saving Strategies**:
- Follow the 50/30/20 rule (needs/wants/savings)
- Use high-yield savings accounts
- Consider CD ladders for short-term goals
- Review and adjust your strategy regularly

Remember: Consistency is key. Even small amounts saved regularly can grow significantly over time."""

        elif 'debt' in prompt_lower or 'loan' in prompt_lower:
            return """**Debt Management Strategy**

Here's a systematic approach to managing your debt:

**Debt Assessment**:
1. List all debts with balances, interest rates, and minimum payments
2. Calculate your total debt-to-income ratio
3. Identify which debts have the highest interest rates

**Debt Repayment Strategies**:
- **Avalanche Method**: Pay off highest interest debt first
- **Snowball Method**: Pay off smallest balances first for motivation
- **Debt Consolidation**: Consider combining multiple debts into one loan

**Immediate Steps**:
1. Stop taking on new debt
2. Pay more than minimum payments when possible
3. Negotiate lower interest rates with creditors
4. Consider balance transfer cards for high-interest debt

**Long-term Prevention**:
- Build emergency fund to avoid future debt
- Live below your means
- Use credit cards responsibly
- Save for major purchases instead of financing

Remember: Getting out of debt takes time and discipline, but it's achievable with a solid plan."""

        else:
            return """**Personal Finance Advice**

Thank you for your question! Here are some general financial principles to consider:

**Financial Foundation**:
1. **Budget**: Track income and expenses to understand your cash flow
2. **Emergency Fund**: Save 3-6 months of expenses for unexpected events
3. **Debt Management**: Prioritize high-interest debt repayment
4. **Insurance**: Protect yourself and your family with appropriate coverage

**Smart Money Habits**:
- Pay yourself first (automate savings)
- Live below your means
- Avoid lifestyle inflation
- Regularly review and adjust your financial plan

**Investment Basics**:
- Start early to benefit from compound interest
- Diversify your investments
- Consider your risk tolerance and time horizon
- Take advantage of employer retirement plans

**Continuous Learning**:
- Stay informed about personal finance topics
- Seek professional advice when needed
- Learn from your financial mistakes
- Set clear, achievable financial goals

Remember: Financial success is a journey, not a destination. Small, consistent actions today will lead to significant results over time."""

    def get_service_status(self) -> Dict[str, bool]:
        """
        Get the status of AI services
        
        Returns:
            Dictionary with service availability status
        """
        return {
            'huggingface_available': self.hf_available,
            'models_loaded': self.models_loaded,
            'models_loading': self.models_loading,
            'sentiment_analyzer_ready': self.sentiment_analyzer is not None,
            'ner_pipeline_ready': self.ner_pipeline is not None,
            'text_generator_ready': self.tokenizer is not None and self.model is not None,
            'fallback_mode': not (self.hf_available and self.models_loaded)
        }
    
    # Specific advice methods for different personas and scenarios
    def _get_student_loan_advice(self, prompt: str) -> str:
        """Provide student-specific advice for loans and debt"""
        if 'gym' in prompt:
            return """**Student Financial Advice: Balancing Gym Membership and Student Loans**

As a student with student loans, here's my advice regarding gym expenses:

**Priority Assessment**:
1. **Student loans first**: These typically have higher interest rates and longer-term impact
2. **Health is important**: But look for cost-effective alternatives

**Smart Fitness Strategies for Students**:
- **University gym**: Use your student recreation center (often included in fees)
- **Budget gyms**: Planet Fitness, LA Fitness student discounts (~$10-15/month)
- **Free alternatives**: YouTube workouts, running, bodyweight exercises
- **Seasonal approach**: Outdoor activities in good weather, gym in winter

**Financial Balance**:
- If gym costs >$30/month, consider alternatives
- Allocate extra income to loan payments first
- Build an emergency fund of $500-1000
- Track all expenses to see where your money goes

**Long-term perspective**: Paying off loans early saves more money than most gym memberships cost. Consider free fitness options during your highest debt period.
"""
        
        return """**Student Loan Management Strategy**

Here's how to handle your student loans effectively:

**Understanding Your Loans**:
1. List all loans with balances, interest rates, servicers
2. Know the difference between federal and private loans
3. Understand your grace period and repayment options

**Repayment Strategies**:
- **Standard repayment**: Highest monthly payment, least interest overall
- **Income-driven plans**: Lower payments based on income (federal loans)
- **Avalanche method**: Pay minimums on all, extra on highest interest rate
- **Snowball method**: Pay smallest balance first for motivation

**Student-Specific Tips**:
- Keep federal loans separate from private (better protections)
- Consider auto-pay discounts (usually 0.25% rate reduction)
- Don't ignore loans - contact servicer if having trouble
- Explore forgiveness programs for public service careers

**While in School**:
- Pay interest on unsubsidized loans if possible
- Avoid borrowing more than necessary
- Look for scholarships and grants continuously

Remember: Student loans are an investment in your future earning potential.
"""
    
    def _get_student_saving_advice(self, prompt: str) -> str:
        """Provide student-specific saving advice"""
        return """**Student Saving Strategies**

Saving money as a student requires creativity and discipline:

**Smart Saving Tactics**:
1. **The $1 rule**: Save every $1 bill you receive
2. **Round-up savings**: Round purchases up, save the difference
3. **Meal prep**: Cook in bulk, avoid dining out frequently
4. **Textbook savings**: Buy used, rent, or use library reserves
5. **Student discounts**: Always ask - many businesses offer them

**Income Opportunities**:
- Work-study jobs on campus
- Tutoring (often pays $15-25/hour)
- Freelance skills (writing, design, coding)
- Paid internships and co-ops
- Sell items you no longer need

**Emergency Fund for Students**:
- Start with $300-500 goal
- Keep it in a separate savings account
- Use only for true emergencies (car repair, medical)

**Long-term Perspective**:
- Build good financial habits now
- Learn to live below your means
- Understand wants vs needs
- Start credit building responsibly

Even saving $25/month as a student builds valuable habits and provides a financial cushion.
"""
    
    def _get_student_budget_advice(self, prompt: str) -> str:
        """Provide student-specific budgeting advice"""
        return """**Student Budgeting Guide**

Creating a budget as a student with irregular income:

**Student Budget Categories**:
- **Fixed costs**: Tuition, rent, insurance, loan payments
- **Variable necessities**: Food, gas, school supplies
- **Discretionary**: Entertainment, dining out, subscriptions

**Budgeting on Irregular Income**:
1. Track income for 3 months to find your average
2. Budget based on your lowest month
3. Save excess from higher-income months
4. Use the envelope method for cash categories

**Student-Specific Tips**:
- **Textbooks**: Budget $400-600/semester, then find savings
- **Food**: Meal plans vs. cooking - calculate the real cost
- **Transportation**: Consider walking/biking vs. car costs
- **Free entertainment**: Campus events, free museum days

**Track These Categories**:
- Housing (aim for <30% of income)
- Food (15-20%)
- Transportation (10-15%)
- School supplies (5-10%)
- Savings (at least 5% even if small amounts)

**Tools**: Use apps like Mint, YNAB (free for students), or simple spreadsheets.

Remember: Your student years are for learning financial discipline that will serve you throughout life.
"""
    
    def _get_student_gym_advice(self, prompt: str) -> str:
        """Specific advice about gym expenses for students"""
        return self._get_student_loan_advice(prompt)  # Reuse the gym-specific advice
    
    def _get_professional_investment_advice(self, prompt: str) -> str:
        """Investment advice for working professionals"""
        return """**Professional Investment Strategy**

As a working professional, here's how to approach investing:

**Investment Priority Order**:
1. **Emergency fund**: 3-6 months expenses in high-yield savings
2. **Employer 401(k) match**: Free money - contribute enough to get full match
3. **High-interest debt**: Pay off credit cards (>6% interest)
4. **Max retirement accounts**: 401(k), IRA ($6,500 limit for 2024)
5. **Taxable investment accounts**: For additional long-term growth

**Asset Allocation by Age**:
- **20s-30s**: 80-90% stocks, 10-20% bonds
- **40s**: 70-80% stocks, 20-30% bonds
- **50s+**: 60-70% stocks, 30-40% bonds
- Rule of thumb: (120 - your age) = % in stocks

**Professional Investment Vehicles**:
- **Index funds**: Low fees, broad diversification (VTI, VXUS)
- **Target-date funds**: Automatic rebalancing
- **ETFs**: Tax-efficient, liquid
- **Real estate**: REITs or rental properties for diversification

**Advanced Strategies**:
- Tax-loss harvesting in taxable accounts
- Backdoor Roth IRA if high income
- Mega backdoor Roth if available
- Consider professional financial advisor for complex situations

Start early, invest consistently, and keep fees low for long-term wealth building.
"""
    
    def _get_professional_retirement_advice(self, prompt: str) -> str:
        """Retirement planning advice for professionals"""
        return """**Professional Retirement Planning**

Maximize your retirement savings as a working professional:

**Retirement Account Limits (2024)**:
- **401(k)**: $23,000 ($30,500 if 50+)
- **IRA**: $7,000 ($8,000 if 50+)
- **Total possible**: $30,000+ annually

**Employer Benefits to Maximize**:
- Get full 401(k) match (typically 3-6% of salary)
- HSA if available ($4,300 individual, $8,550 family)
- Consider after-tax 401(k) contributions if mega backdoor available

**Professional Retirement Strategies**:
1. **Automate everything**: Set up automatic contributions
2. **Increase with raises**: Boost contribution % with each promotion
3. **Tax diversification**: Mix of traditional and Roth accounts
4. **Rebalance annually**: Maintain target asset allocation

**Income-Specific Considerations**:
- **High earners**: May be limited from Roth IRA (backdoor option)
- **Variable income**: Contribute heavily in high-earning years
- **Stock options**: Understand vesting and tax implications

**Retirement Planning Milestones**:
- Age 30: 1x annual salary saved
- Age 40: 3x annual salary
- Age 50: 6x annual salary
- Age 60: 8x annual salary
- Retirement: 10-12x annual salary

Consider working with a fee-only financial advisor for comprehensive planning.
"""
    
    def _get_professional_tax_advice(self, prompt: str) -> str:
        """Tax optimization advice for professionals"""
        return """**Professional Tax Optimization**

Strategies to minimize your tax burden legally:

**Pre-Tax Savings (Reduces Current Taxes)**:
- 401(k) contributions
- Traditional IRA (if eligible)
- HSA contributions (triple tax advantage)
- Flexible Spending Account (FSA)

**Tax Credits vs. Deductions**:
- **Credits**: Dollar-for-dollar tax reduction (Child Tax Credit, Education Credits)
- **Deductions**: Reduce taxable income (mortgage interest, charitable donations)

**Professional Tax Strategies**:
1. **Maximize retirement contributions**: Immediate tax savings
2. **Tax-loss harvesting**: Offset gains with losses in taxable accounts
3. **Charitable giving**: Bunching donations, donor-advised funds
4. **Professional development**: Many education expenses are deductible

**Advanced Strategies**:
- **Backdoor Roth**: If income too high for regular Roth
- **Mega backdoor Roth**: If employer plan allows
- **Tax-efficient fund placement**: Bonds in tax-advantaged accounts
- **Municipal bonds**: For high earners in high-tax states

**Business Owners Additional Options**:
- SEP-IRA or Solo 401(k)
- Business expense deductions
- Qualified Business Income (QBI) deduction

**Important**: Tax laws change frequently. Consider consulting a tax professional for complex situations or significant income changes.

Proactive tax planning can save thousands annually.
"""
    
    def _get_general_saving_advice(self, prompt: str) -> str:
        """General saving advice for all personas"""
        return self._fallback_response_generation(prompt, "general")
    
    def _get_debt_management_advice(self, prompt: str) -> str:
        """General debt management advice"""
        return self._fallback_response_generation(prompt, "general")
    
    def _get_investment_advice(self, prompt: str) -> str:
        """General investment advice"""
        return self._fallback_response_generation(prompt, "general")
    
    def _get_budget_advice(self, prompt: str) -> str:
        """General budget advice"""
        return self._fallback_response_generation(prompt, "general")
    
    def _get_emergency_fund_advice(self, prompt: str) -> str:
        """Emergency fund advice"""
        return """**Emergency Fund Essentials**

Building your financial safety net:

**Emergency Fund Basics**:
- **Purpose**: Cover unexpected expenses without debt
- **Amount**: 3-6 months of essential expenses
- **Location**: High-yield savings account (accessible but separate)

**How Much Do You Need?**:
- **Single, stable job**: 3-4 months expenses
- **Married, dual income**: 3-4 months expenses
- **Single earner household**: 6 months expenses
- **Variable income**: 6+ months expenses
- **High-risk job**: 6+ months expenses

**Building Your Emergency Fund**:
1. **Start small**: Even $500 helps with minor emergencies
2. **Automate**: Set up automatic weekly/monthly transfers
3. **Use windfalls**: Tax refunds, bonuses, gifts
4. **Side income**: Temporarily direct extra earnings to fund

**What Qualifies as an Emergency?**:
✅ **True emergencies**: Job loss, medical bills, major car/home repairs
❌ **Not emergencies**: Vacations, shopping, known upcoming expenses

**Where to Keep It**:
- High-yield savings account (current rates 4-5%)
- Money market account
- Short-term CDs (if you have multiple months saved)
- NOT: Checking account, investment accounts, cash at home

**After It's Built**:
- Only use for true emergencies
- Replenish immediately after use
- Review annually and adjust for lifestyle changes

Your emergency fund provides peace of mind and financial stability.
"""
