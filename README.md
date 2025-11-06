# Personal Finance Chatbot

An intelligent conversational AI system that provides personalized financial guidance for savings, taxes, and investments. Built with Streamlit, FastAPI, and Hugging Face models.

## 🎯 Project Overview

This Personal Finance Chatbot leverages multiple AI services and models to provide:

- **Personalized Financial Guidance** - Customized advice based on user profiles
- **AI-Generated Budget Summaries** - Automatic budget analysis and recommendations
- **Spending Insights and Suggestions** - Actionable recommendations for expense optimization
- **Demographic-Aware Communication** - Adapts tone based on user type (student vs professional)
- **Advanced NLP Analysis** - Real sentiment analysis, entity recognition, and keyword extraction
- **Hybrid AI Architecture** - Multiple AI backends with fallback mechanisms for reliability
- **Conversational NLP Experience** - Natural, context-aware interactions

## 🏗️ Architecture

### Enhanced Hybrid AI Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   AI Service    │    │   Multiple AI   │
│   Frontend      │◄──►│   Backend       │◄──►│   Layer         │◄──►│   Backends      │
│                 │    │                 │    │                 │    │                 │
│ • Chat UI       │    │ • API Endpoints │    │ • Smart Routing │    │ • Hugging Face  │
│ • Forms         │    │ • Data Processing│   │ • Fallback Logic│    │ • Local Models  │
│ • Visualizations│    │ • Validation    │    │ • Response Mgmt │    │ • Fallback NLP  │
│ • Charts        │    │ • Error Handling│    │ • Caching       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### AI Service Integration Options

The system supports an open-source-first approach with automatic fallback:

1. **Primary: Hugging Face Models**
   - Open-source transformer models (BERT, RoBERTa, GPT-2, etc.)
   - Local or cloud-based inference
   - Cost-effective and flexible

2. **Fallback: Rule-based System**
   - Keyword extraction
   - Template-based responses
   - Always available backup

## 🚀 Features

### 1. General Chat Interface
- Ask any financial question
- Get personalized advice based on your profile
- View sentiment analysis and keyword extraction
- Support for different personas (student, professional, general)

### 2. Budget Summary Analysis
- Comprehensive budget breakdown
- Financial health assessment
- Top spending category identification
- Personalized recommendations
- Visual expense distribution charts

### 3. Spending Insights
- Deep spending pattern analysis
- Goal feasibility assessment
- Behavioral insights
- Actionable recommendations
- Category-wise spending breakdown

### 4. Text Analysis
- Sentiment analysis of financial text
- Keyword extraction
- Entity recognition
- Category classification
- Interactive visualizations

## 📋 Prerequisites

- Python 3.8 or higher
- Basic understanding of personal finance concepts

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd personal-finance-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```bash
# Hugging Face Configuration
HUGGINGFACE_TOKEN=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

# Model Configuration
TEXT_GENERATION_MODEL_NAME=ibm-granite/granite-speech-3.3-8b
NLU_MODEL_NAME=bert-base-uncased-finetuned-sst-2-english
NER_MODEL_NAME=dslim/bert-base-NER

# Application Settings
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
```

### 5. Hugging Face Setup

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Generate an access token from your account settings
3. Add the token and model names to your `.env` as shown above

#### Benefits:
- **Cost-effective**: Use open-source models
- **Local processing**: Run models locally for privacy
- **Flexibility**: Choose from thousands of pre-trained models
- **Fallback option**: Automatic fallback when models are unavailable

#### Testing Hugging Face Setup:
```bash
python test_huggingface.py
```

For detailed setup instructions, see `HUGGINGFACE_SETUP.md`.

## 🚀 Running the Application

### 1. Start the Backend Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend Application
```bash
streamlit run streamlit_app.py
```

### 3. Access the Application
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📖 Usage Guide

### General Chat
1. Navigate to the "Chat" section
2. Select your profile type (Student, Professional, or General)
3. Type your financial question
4. Click "Send" to get personalized advice
5. View the analysis details in the expandable section

### Budget Summary
1. Go to the "Budget Summary" section
2. Enter your monthly income and savings goal
3. Fill in your monthly expenses
4. Select your profile type
5. Click "Analyze Budget" to get comprehensive insights
6. Review the charts and recommendations

### Spending Insights
1. Navigate to "Spending Insights"
2. Enter your income and financial goals
3. Provide detailed expense breakdown
4. Select your profile type
5. Click "Analyze Spending" for behavioral insights
6. Review action items and recommendations

### Text Analysis
1. Go to "Text Analysis"
2. Enter any financial text
3. Click "Analyze Text"
4. View sentiment, keywords, entities, and categories

## 🔧 API Endpoints

### Chat Endpoint
```http
POST /chat
{
  "message": "How can I save money?",
  "persona": "student"
}
```

### Budget Summary Endpoint
```http
POST /budget-summary
{
  "income": 5000.0,
  "expenses": {
    "rent": 1500.0,
    "groceries": 400.0
  },
  "savings_goal": 500.0,
  "persona": "professional"
}
```

### Spending Insights Endpoint
```http
POST /spending-insights
{
  "income": 5000.0,
  "expenses": {
    "rent": 1500.0,
    "groceries": 400.0
  },
  "goals": ["emergency fund", "vacation"],
  "persona": "general"
}
```

### NLU Analysis Endpoint
```http
POST /nlu
{
  "text": "I'm struggling with my budget"
}
```

## 🎨 Customization

### Adding New Personas
1. Modify the `build_basic_prompt()` function in `utils.py`
2. Add persona-specific logic in the prompt building functions
3. Update the frontend persona selection options

### Customizing Prompts
1. Edit the prompt templates in `utils.py`
2. Adjust the tone and complexity based on your needs
3. Test with different user inputs

### Adding New Features
1. Create new API endpoints in `main.py`
2. Add corresponding frontend interfaces in `streamlit_app.py`
3. Update the utility functions as needed

## 🔍 Troubleshooting

### Common Issues

#### Backend Not Starting
- Check if port 8000 is available
- Verify all dependencies are installed
- Check environment variables are set correctly

#### Frontend Connection Issues
- Ensure backend is running on http://localhost:8000
- Check CORS settings in `main.py`
- Verify API health endpoint is responding

#### AI Service Errors
- Verify IBM Watson credentials are correct
- Check API quotas and limits
- Review error logs for specific issues

#### Missing Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 Performance Optimization

### For Production Deployment
1. Use production-grade WSGI server (Gunicorn)
2. Implement caching for AI responses
3. Add rate limiting for API endpoints
4. Use environment-specific configurations
5. Implement proper error handling and logging

### Scaling Considerations
1. Use load balancers for multiple instances
2. Implement database for user sessions
3. Add monitoring and analytics
4. Consider using Redis for caching

## 🤝 Contributing

We welcome contributions from the community! This project is designed to be contributor-friendly with a modular architecture and comprehensive documentation.

### 🚀 Getting Started for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/personal-finance-chatbot.git
   cd personal-finance-chatbot
   ```
3. **Set up the development environment** following the installation instructions above
4. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### 📝 Contribution Guidelines

#### Code Style and Standards
- **Follow PEP 8** for Python code formatting
- **Use type hints** for all function parameters and return values
- **Write docstrings** for all functions and classes
- **Keep functions focused** and single-purpose
- **Use meaningful variable names**

#### Architecture Guidelines
- **Maintain separation of concerns** between frontend, backend, and AI services
- **Add fallback mechanisms** for any new AI service integrations
- **Include error handling** for all external API calls
- **Follow the existing modular structure**
- **Update documentation** for any architectural changes

#### Testing Requirements
- **Add tests** for all new features and bug fixes
- **Ensure existing tests pass** before submitting PR
- **Test both success and error scenarios**
- **Run the full test suite**:
  ```bash
  python test_app.py
  python test_huggingface.py  # If using Hugging Face features
  ```

#### Documentation Requirements
- **Update README.md** for new features or setup changes
- **Add inline code comments** for complex logic
- **Update API documentation** for new endpoints
- **Include examples** for new functionality
- **Update IMPLEMENTATION_GUIDE.md** for architectural changes

### 🎯 Areas for Contribution

#### 🔥 High Priority
- **Enhanced AI Models**: Integration with additional AI services (OpenAI, Anthropic, etc.)
- **Real Financial Data**: Integration with banking APIs or financial data providers
- **Advanced Analytics**: More sophisticated financial analysis and insights
- **Mobile Responsiveness**: Better mobile UI/UX
- **Performance Optimization**: Caching, database integration, response times

#### 🔄 Medium Priority
- **New Personas**: Additional user types (retirees, small business owners, etc.)
- **Localization**: Multi-language support
- **Visualization Enhancements**: Better charts and interactive elements
- **Security Improvements**: Authentication, data encryption, audit trails
- **Testing Coverage**: Expand test coverage and add integration tests

#### 📦 Good for Beginners
- **UI/UX Improvements**: CSS styling, better form layouts
- **Documentation**: Improve existing docs, add tutorials
- **Bug Fixes**: Fix issues reported in GitHub issues
- **Code Refactoring**: Clean up code, improve readability
- **Configuration**: Environment setup improvements

### 🛠️ Development Setup

#### Running Tests
```bash
# Run main application tests
python test_app.py

# Test AI model integrations
python test_huggingface.py
python test_ai_models.py

# Test direct API calls
python test_api_direct.py
```

#### Development Mode
```bash
# Backend with hot reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend with auto-refresh
streamlit run streamlit_app.py
```

#### Code Formatting
```bash
# Install development dependencies
pip install black flake8 mypy

# Format code
black *.py

# Check style
flake8 *.py

# Type checking
mypy *.py
```

### 📝 Pull Request Process

1. **Ensure your code follows** the style guidelines
2. **Add or update tests** for your changes
3. **Update documentation** as needed
4. **Test your changes thoroughly**:
   - Run the full test suite
   - Test the UI manually
   - Verify API endpoints work correctly
5. **Create a descriptive PR title** and description:
   - Explain what the change does
   - Reference any related issues
   - Include screenshots for UI changes
   - List any breaking changes
6. **Request review** from maintainers

### 📚 Resources for Contributors

#### Project Documentation
- **IMPLEMENTATION_GUIDE.md**: Detailed code architecture explanation
- **HUGGINGFACE_SETUP.md**: AI model integration guide
- **API Documentation**: Available at `/docs` when backend is running
- **Test Files**: Multiple test files for different components

#### Learning Resources
- **FastAPI Tutorial**: https://fastapi.tiangolo.com/tutorial/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Hugging Face Documentation**: https://huggingface.co/docs
- **Transformers Library**: https://huggingface.co/docs/transformers

## 🆕 Recent Updates & Changes

### ✨ Latest Features (August 2025)

#### 🤖 Enhanced AI Integration
- **Hugging Face Support**: Added full integration with Hugging Face transformers
- **Multiple AI Backends**: Smart routing between Hugging Face models and fallback systems
- **Real NLP Models**: Actual sentiment analysis, entity recognition, and keyword extraction
- **Cost Optimization**: Use free open-source models as alternatives

#### 🏢 Improved Architecture
- **Hybrid AI System**: Multi-layered AI service with automatic failover
- **Enhanced Error Handling**: Graceful degradation when services are unavailable
- **Better Performance**: Optimized response times and caching
- **Modular Design**: Easier to add new AI services and features

#### 🛠️ Development Improvements
- **Comprehensive Testing**: Multiple test suites for different components
- **Better Documentation**: Detailed setup guides and implementation explanations
- **Development Tools**: Hot reload, automated testing, code formatting
- **Contributor Guidelines**: Clear processes for community contributions

#### 🔒 Security & Reliability
- **Environment Security**: Better secrets management with `.env` files
- **Input Validation**: Enhanced data validation and sanitization
- **Version Control**: Added `.gitignore` for secure development
- **Fallback Mechanisms**: Always-available basic functionality

### 💯 What's Changed Since Initial Release

1. **AI Service Layer**: Completely redesigned with multiple backend support
2. **Dependencies**: Updated to include Hugging Face transformers and advanced NLP libraries
3. **Testing**: Added comprehensive test suite with multiple test files
4. **Documentation**: Enhanced with detailed guides and contribution instructions
5. **Configuration**: Improved setup process with better environment management
6. **Error Handling**: More robust error handling and fallback systems
7. **Performance**: Optimized for better response times and resource usage

### 🚀 Upgrade Guide

If you're upgrading from an earlier version:

1. **Update Dependencies**: Run `pip install -r requirements.txt` to get new packages
2. **Update Environment**: Add new environment variables (see `.env` example)
3. **Test Integration**: Run the test suite to ensure everything works
4. **Review Documentation**: Check new features in `HUGGINGFACE_SETUP.md`
5. **Optional Setup**: Configure Hugging Face integration if desired

### 🔎 What's Next

Upcoming features and improvements:
- Integration with additional AI providers (OpenAI, Anthropic)
- Real-time financial data integration
- Advanced analytics dashboard
- Mobile-responsive UI improvements
- Database integration for user sessions

### 🚑 Reporting Issues

When reporting bugs or requesting features:

1. **Search existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Error messages and logs
4. **Add labels** to categorize the issue

### 🏆 Recognition

Contributors will be:
- **Listed in the README** acknowledgments section
- **Mentioned in release notes** for significant contributions
- **Invited to join** the project maintainer team for outstanding contributions

### 💬 Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Request Reviews**: For code-related discussions

We strive to be responsive and supportive to all contributors. Don't hesitate to ask questions or request help!

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- IBM Watson AI services for natural language processing
- Streamlit for the user interface framework
- FastAPI for the backend API framework
- The open-source community for various libraries and tools

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at http://localhost:8000/docs

## 🔮 Future Enhancements

- Integration with real financial data sources
- Mobile app development
- Advanced analytics and reporting
- Multi-language support
- Voice interface integration
- Machine learning model fine-tuning
- Integration with financial institutions (with user consent)
- Real-time market data integration
- Goal tracking and progress monitoring
- Social features for financial communities

---

**Note**: This chatbot provides educational financial advice and should not be considered as professional financial consultation. Always consult with qualified financial advisors for personalized financial planning.
