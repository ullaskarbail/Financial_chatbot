# Hugging Face Integration with WatsonX Models

This guide explains how to set up and use Hugging Face with WatsonX models for the Personal Finance Chatbot.

## Prerequisites

1. **Hugging Face Account**: Create an account at [huggingface.co](https://huggingface.co)
2. **Access Token**: Generate an access token from your Hugging Face account settings
3. **Python Environment**: Ensure you have Python 3.8+ with the required dependencies

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# Hugging Face Configuration
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Model Configuration (via Hugging Face)
WATSONX_MODEL_NAME=distilgpt2
NLU_MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest
NER_MODEL_NAME=dslim/bert-base-NER

# Application Settings
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
```

### 3. Get Your Hugging Face Token

1. Go to [huggingface.co](https://huggingface.co) and sign in
2. Click on your profile picture → Settings
3. Go to "Access Tokens" in the left sidebar
4. Click "New token"
5. Give it a name (e.g., "Personal Finance Chatbot")
6. Select "Read" permissions
7. Copy the generated token and paste it in your `.env` file

### 4. Model Access

The following models are used:

- **Text Generation**: `distilgpt2` - For text generation (currently used for NLU only)
- **Sentiment Analysis**: `cardiffnlp/twitter-roberta-base-sentiment-latest` - For sentiment analysis
- **Named Entity Recognition**: `dslim/bert-base-NER` - For entity extraction

These models are publicly available on Hugging Face Hub.

## Testing the Integration

Run the test script to verify everything is working:

```bash
python test_huggingface.py
```

This will:
1. Initialize the AI service
2. Test NLU analysis with a sample text
3. Test response generation with a sample prompt
4. Display the results

## Features

### NLU Analysis (Using Hugging Face Models)
- **Sentiment Analysis**: Determines if text is positive, negative, or neutral using real AI models
- **Keyword Extraction**: Identifies important financial terms
- **Entity Recognition**: Extracts named entities (people, organizations, etc.)
- **Category Classification**: Categorizes text into financial topics

### Response Generation (Hybrid Approach)
- **Current Implementation**: Uses a sophisticated financial knowledge base with persona-based responses
- **Future Enhancement**: Will integrate with specialized financial AI models
- **Persona-based Responses**: Tailored advice for different user types (student, professional, general)
- **Context-aware**: Considers the user's financial situation and goals
- **Actionable Advice**: Provides specific, implementable recommendations

## Current Architecture

The system uses a hybrid approach:

1. **NLU Analysis**: Real Hugging Face models for sentiment analysis, entity recognition, and keyword extraction
2. **Response Generation**: Sophisticated financial knowledge base with rule-based responses that are more accurate and actionable than general-purpose AI models

This approach ensures:
- **Accurate Financial Advice**: Responses are based on proven financial principles
- **Real AI Analysis**: User input is analyzed using state-of-the-art NLP models
- **Consistent Quality**: Responses are always helpful and relevant
- **Fast Performance**: No need to wait for large language model generation

## Troubleshooting

### Common Issues

1. **"Hugging Face token not found"**
   - Ensure your `.env` file contains the `HUGGINGFACE_TOKEN`
   - Verify the token is valid and has read permissions

2. **"Error loading models"**
   - Check your internet connection
   - Ensure you have sufficient disk space for model downloads
   - Verify the model names are correct

3. **"CUDA out of memory"**
   - The models will automatically use CPU if GPU memory is insufficient
   - Consider using smaller models if needed

### Performance Notes

- **First Run**: Models will be downloaded on first use (may take several minutes)
- **Memory Usage**: Models require significant RAM (4-8GB recommended)
- **Response Time**: Initial model loading takes time, but subsequent requests are faster

## Model Customization

You can customize the models by changing the environment variables:

```env
# Use different text generation model
WATSONX_MODEL_NAME=ibm-granite/granite-speech-3.3-8b

# Use different sentiment model
NLU_MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment

# Use different NER model
NER_MODEL_NAME=dbmdz/bert-large-cased-finetuned-conll03-english
```

## Security Notes

- Keep your Hugging Face token secure and never commit it to version control
- The token only needs read permissions for model access
- Models are downloaded locally and cached for future use

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify your environment configuration
3. Test with the provided test script
4. Check Hugging Face Hub for model availability and documentation
