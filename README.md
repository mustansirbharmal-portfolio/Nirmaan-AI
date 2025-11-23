# 🎯 Nirmaan AI Communication Scorer

An advanced AI-powered platform for analyzing and scoring students' spoken communication skills. This application combines rule-based methods, NLP-based semantic scoring, and data-driven rubric evaluation to provide comprehensive feedback on self-introduction presentations.

## ✨ Key Features

- **🎤 Web Audio Recording**: Record directly in browser with real-time processing
- **🧠 AI-Powered Analysis**: Azure OpenAI integration for advanced text analysis
- **📊 Multi-Criteria Scoring**: 5 comprehensive evaluation criteria with weighted scoring
- **💬 Detailed Feedback**: Actionable insights with examples and improvement suggestions
- **🔍 Similarity Search**: Find similar transcripts using vector embeddings
- **☁️ Azure Integration**: Complete cloud-based processing pipeline
- **📱 Modern UI**: Responsive design with professional interface
- **🔄 Real-time Processing**: Instant feedback and analysis

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.8+** installed on your system
- **Git** for cloning the repository
- **Azure Account** with active subscriptions for:
  - Azure OpenAI Service
  - Azure Speech Services
  - Azure Cosmos DB
  - Azure Storage Account

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Nirmaan
```

### Step 2: Set Up Python Environment
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Azure Services
The `.env` file contains all necessary Azure configurations. Ensure your Azure services are set up:

```env
# Azure Speech Services
AZURE_SPEECH_KEY=your_speech_service_key
AZURE_SPEECH_REGION=your_region

# Azure Cosmos DB
COSMOS_ENDPOINT=https://your-account.documents.azure.com:443/
COSMOS_KEY=your_cosmos_key
COSMOS_DATABASE_NAME=database_name
COSMOS_CONTAINER_NAME=container_name

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Azure Storage (for audio recordings)
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
AZURE_STORAGE_CONTAINER=container-name
```

### Step 4: Run the Application
```bash
python run.py
```

The application will start on **http://localhost:5000**

### Step 5: Test the Application
1. Open your browser and navigate to `http://localhost:5000`
2. Click **"Load Sample Data"** to test with provided sample
3. Click **"Score Communication"** to see the analysis
4. Try the **audio recording feature** by clicking the microphone icon

## 📊 Detailed Scoring Formula

### Overall Score Calculation
```
Overall Score = (Content×40% + Speech Rate×10% + Language×20% + Clarity×15% + Engagement×15%)
```

### 1. Content & Structure Analysis (40 points max)

#### Salutation Scoring (5 points max)
```python
def score_salutation(text):
    greetings = ["hello", "hi", "good morning", "good afternoon", "good evening", "namaste"]
    
    if any(greeting in text.lower()[:100] for greeting in greetings):
        return 5  # Excellent greeting
    elif any(greeting in text.lower() for greeting in greetings):
        return 3  # Greeting present but not at start
    else:
        return 0  # No greeting found
```

#### Keyword Presence Scoring (30 points max)
```python
def score_keywords(text):
    must_have_keywords = {
        "name_introduction": ["name", "myself", "i am", "i'm"],      # 8 points
        "age": ["years old", "age", "year old"],                    # 4 points
        "education": ["school", "class", "grade", "studying"],      # 6 points
        "family": ["family", "parents", "mother", "father"],        # 4 points
        "interests": ["like", "enjoy", "hobby", "interest"],        # 4 points
        "personal_trait": ["special", "unique", "about me"],        # 2 points
        "closing": ["thank you", "thanks", "pleasure"]              # 2 points
    }
    
    total_score = 0
    for category, keywords in must_have_keywords.items():
        if any(keyword in text.lower() for keyword in keywords):
            total_score += category_points[category]
    
    return min(30, total_score)
```

#### Flow Order Scoring (5 points max)
```python
def score_flow(text):
    # Check if introduction starts with proper greeting
    if any(greeting in text.lower()[:50] for greeting in ["hi", "hello", "good"]):
        return 5
    else:
        return 0
```

### 2. Speech Rate Analysis (10 points max)
```python
def score_speech_rate(word_count, duration_seconds):
    if not duration_seconds:
        return 5  # Default score if no duration provided
    
    wpm = (word_count / duration_seconds) * 60
    
    if 111 <= wpm <= 140:      # Optimal range
        return 10
    elif 90 <= wpm < 111:      # Slightly slow
        return 8
    elif 140 < wpm <= 180:     # Slightly fast
        return 7
    elif 70 <= wpm < 90:       # Too slow
        return 5
    elif 180 < wpm <= 220:     # Too fast
        return 4
    else:                      # Extremely slow/fast
        return 2
```

### 3. Language & Grammar Analysis (20 points max)

#### Vocabulary Richness - TTR (10 points max)
```python
def calculate_ttr(text):
    words = text.lower().split()
    unique_words = set(words)
    ttr = len(unique_words) / len(words) if words else 0
    
    if ttr >= 0.9:      return 10  # Excellent variety
    elif ttr >= 0.7:    return 8   # Good variety
    elif ttr >= 0.5:    return 6   # Average variety
    elif ttr >= 0.3:    return 4   # Limited variety
    else:               return 2   # Poor variety
```

#### Grammar Quality (10 points max)
```python
def score_grammar(text):
    # Simplified grammar scoring (can be enhanced with language_tool_python)
    grammar_issues = 0
    
    # Check common errors
    if "i are" in text.lower(): grammar_issues += 1
    if "he are" in text.lower(): grammar_issues += 1
    if "she are" in text.lower(): grammar_issues += 1
    
    # Article usage
    if any(phrase in text.lower() for phrase in ["i am student", "i work in company"]):
        grammar_issues += 1
    
    # Score based on issues found
    if grammar_issues == 0:     return 10
    elif grammar_issues == 1:   return 8
    elif grammar_issues == 2:   return 6
    else:                       return 4
```

### 4. Clarity Analysis (15 points max)
```python
def score_clarity(text):
    filler_words = ["um", "uh", "like", "you know", "actually", "basically", "literally"]
    
    word_count = len(text.split())
    filler_count = sum(text.lower().count(filler) for filler in filler_words)
    filler_rate = (filler_count / word_count) * 100 if word_count > 0 else 0
    
    if filler_rate <= 6:        return 15  # Excellent clarity
    elif filler_rate <= 9:      return 12  # Good clarity
    elif filler_rate <= 12:     return 9   # Fair clarity
    elif filler_rate <= 15:     return 6   # Poor clarity
    else:                       return 3   # Very poor clarity
```

### 5. Engagement Analysis (15 points max)
```python
def score_engagement(text):
    # Using Azure OpenAI for sentiment analysis
    sentiment_score = analyze_sentiment_with_azure(text)  # Returns 0.0-1.0
    
    if sentiment_score >= 0.9:      return 15  # Highly positive
    elif sentiment_score >= 0.7:    return 12  # Positive
    elif sentiment_score >= 0.5:    return 9   # Neutral
    elif sentiment_score >= 0.3:    return 6   # Slightly negative
    else:                           return 3   # Negative
```

## 🎯 Scoring Examples

### Example 1: High-Scoring Introduction (Score: 86/100)
```
"Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School. 
I am 13 years old. I live with my family. There are 3 people in my family, me, my mother 
and my father. One special thing about my family is that they are very kind hearted to 
everyone and soft spoken. One thing I really enjoy is playing cricket and taking wickets. 
A fun fact about me is that I see movies and talk by myself. One thing people don't know 
about me is that I once stole a toy from one of my cousin. My favourite subject is science 
because it is very interesting. Through science I can explore the whole world and make 
discoveries and improve the lives of others. Thank you for listening."
```

**Score Breakdown:**
- Content & Structure: 35/40 (Good greeting, all key elements present)
- Speech Rate: 10/10 (138.9 WPM - optimal range)
- Language & Grammar: 18/20 (Good vocabulary, minor grammar issues)
- Clarity: 12/15 (Few filler words, clear speech)
- Engagement: 15/15 (Positive sentiment, enthusiastic tone)

### Example 2: Areas for Improvement (Score: 45/100)
```
"Um, hi, I'm John. I like, you know, playing games and stuff. That's basically it."
```

**Score Breakdown:**
- Content & Structure: 8/40 (Basic greeting, missing key information)
- Speech Rate: 5/10 (Too fast delivery)
- Language & Grammar: 8/20 (Limited vocabulary, simple sentences)
- Clarity: 3/15 (High filler word rate)
- Engagement: 6/15 (Neutral sentiment, low enthusiasm)

## 🛠️ Technology Stack

### Backend Technologies
- **Python 3.8+**: Core programming language
- **Flask 2.3.3**: Lightweight web framework
- **Azure OpenAI**: GPT-4 for advanced text analysis and embeddings
- **Azure Speech Services**: Real-time speech-to-text conversion
- **Azure Cosmos DB**: NoSQL database for embeddings storage
- **Azure Storage**: Blob storage for audio recordings
- **Sentence Transformers**: Local semantic similarity analysis
- **NumPy**: Numerical computations and vector operations

### Frontend Technologies
- **HTML5**: Modern semantic markup
- **JavaScript ES6+**: Interactive functionality and Web Audio API
- **Bootstrap 5**: Responsive UI framework
- **Font Awesome**: Professional icon library
- **Web Audio API**: Browser-based audio recording

### Azure Cloud Services
- **Azure OpenAI Service**: text-embedding-ada-002 and GPT-4
- **Azure Speech Services**: Speech-to-text with multiple format support
- **Azure Cosmos DB**: Vector embeddings and metadata storage
- **Azure Blob Storage**: Audio file storage and management

## 📋 Complete Setup Instructions

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux Ubuntu 18.04+
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space
- **Internet**: Stable connection for Azure services

### Detailed Installation Steps

#### 1. Environment Preparation
```bash
# Check Python version
python --version  # Should be 3.8+

# Update pip
python -m pip install --upgrade pip

# Install virtualenv if not available
pip install virtualenv
```

#### 2. Project Setup
```bash
# Clone repository
git clone <repository-url>
cd Nirmaan

# Create and activate virtual environment
python -m venv nirmaan-env

# Windows activation:
nirmaan-env\Scripts\activate

# macOS/Linux activation:
source nirmaan-env/bin/activate

# Verify activation (should show virtual env path)
which python
```

#### 3. Dependencies Installation
```bash
# Install all required packages
pip install -r requirements.txt

# Verify key packages
pip list | grep -E "(flask|azure|numpy|torch)"
```

#### 4. Azure Services Configuration

**Required Azure Resources:**
1. **Azure OpenAI Service**
   - Deploy GPT-4 model
   - Deploy text-embedding-ada-002 model
   - Note endpoint and API key

2. **Azure Speech Services**
   - Create Speech resource
   - Note API key and region

3. **Azure Cosmos DB**
   - Create Cosmos DB account (SQL API)
   - Create database
   - Create container
   - Note endpoint and primary key

4. **Azure Storage Account**
   - Create storage account
   - Create container: `files`
   - Note connection string

#### 5. Environment Variables Setup
Update the `.env` file with your Azure credentials:

```env
# Copy from Azure Speech Services
AZURE_SPEECH_KEY=your_actual_speech_key_here
AZURE_SPEECH_REGION=eastus  # or your region

# Copy from Azure Cosmos DB
COSMOS_ENDPOINT=https://your-cosmosdb-account.documents.azure.com:443/
COSMOS_KEY=your_actual_cosmos_primary_key_here
COSMOS_DATABASE_NAME=database_name
COSMOS_CONTAINER_NAME=container_name

# Copy from Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_actual_openai_key_here
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o  # your GPT-4 deployment name
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Copy from Azure Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_STORAGE_CONTAINER=files
```

#### 6. Application Launch
```bash
# Start the application
python run.py

# Expected output:
# INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: all-MiniLM-L6-v2
# INFO:app:Sentence transformer model loaded successfully
# * Running on http://127.0.0.1:5000
```

#### 7. Verification Steps
1. **Open Browser**: Navigate to `http://localhost:5000`
2. **Test Sample**: Click "Load Sample Data" → "Score Communication"
3. **Test Recording**: Click microphone icon and record a short message
4. **Check Logs**: Verify no error messages in console


## 🌐 Usage Guide

### Basic Workflow
1. **Text Input**: Enter or paste self-introduction text
2. **Duration** (Optional): Specify speech duration for rate analysis
3. **Analysis**: Click "Score Communication" for comprehensive analysis
4. **Results**: View overall score and detailed feedback
5. **Improvements**: Follow specific suggestions for enhancement

### Audio Recording Workflow
1. **Start Recording**: Click microphone button
2. **Speak Clearly**: Record your self-introduction (max 2 minutes)
3. **Stop Recording**: Click stop button
4. **Auto-Processing**: System automatically:
   - Stores audio in Azure Storage
   - Converts speech to text
   - Generates vector embeddings
   - Provides comprehensive analysis

### Advanced Features
- **Similarity Search**: Find similar introductions in database
- **Sample Data**: Test with provided high-quality examples
- **Detailed Feedback**: Expandable sections with specific improvements
- **Export Results**: Copy analysis for external use


## Scoring Algorithm

The application uses a multi-layered approach:

1. **Rule-based Scoring**: Keyword matching, word count validation, pattern recognition
2. **NLP-based Scoring**: Semantic similarity using sentence transformers
3. **AI-enhanced Analysis**: Azure OpenAI for sentiment analysis and advanced text understanding
4. **Vector Embeddings**: Azure OpenAI text-embedding-ada-002 for semantic representation
5. **Weighted Combination**: Final score calculated using rubric weights

## Vector Embeddings & Similarity Search

### Embeddings Generation
- **Azure OpenAI Integration**: Uses `text-embedding-ada-002` model for high-quality embeddings
- **Automatic Processing**: Every transcript automatically generates embeddings during scoring
- **Metadata Storage**: Stores additional context like student name, school, timestamp



### Use Cases
1. **Plagiarism Detection**: Identify similar or duplicate submissions
2. **Performance Benchmarking**: Compare against historical submissions
3. **Content Analysis**: Find patterns in communication styles
4. **Recommendation System**: Suggest improvement examples based on similar high-scoring transcripts

## File Structure

```
Nirmaan/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (Azure configs)
├── README.md             # This file
├── static/
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── js/
│       └── app.js        # Frontend JavaScript
└── templates/
    └── index.html        # Main HTML template
```

## Development

### Adding New Criteria
1. Update `RUBRIC_CONFIG` in `app.py`
2. Implement scoring logic in `CommunicationScorer` class
3. Update frontend display in `app.js`

### Customizing Weights
Modify the weights in `RUBRIC_CONFIG` to adjust criterion importance.

### Extending Analysis
Add new analysis methods to the `CommunicationScorer` class and integrate with the main scoring function.

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment

#### Deploy on Render (Recommended)

**Step 1: Prepare Repository**
1. Push your code to GitHub/GitLab
2. Ensure all files are committed:
   - `render.yaml` (deployment configuration)
   - `requirements.txt` (dependencies)
   - `run.py` (WSGI entry point)
   - `.renderignore` (exclude unnecessary files)

**Step 2: Create Render Service**
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub/GitLab repository
4. Select the `Nirmaan` repository

**Step 3: Configure Service**
- **Name**: `nirmaan-ai-scorer`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT run:app`
- **Plan**: Free (or paid for better performance)

**Step 4: Set Environment Variables**
Add these environment variables in Render dashboard:

```env
FLASK_ENV=production
AZURE_SPEECH_KEY=your_speech_service_key
AZURE_SPEECH_REGION=your_region
COSMOS_ENDPOINT=https://your-account.documents.azure.com:443/
COSMOS_KEY=your_cosmos_key
COSMOS_DATABASE_NAME=database_name
COSMOS_CONTAINER_NAME=container_name
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
AZURE_STORAGE_CONTAINER=container-name
```

**Step 5: Deploy**
1. Click "Create Web Service"
2. Render will automatically build and deploy
3. Your app will be available at: `https://your-service-name.onrender.com`

#### Alternative: Manual Deployment
```bash
# Set production environment
export FLASK_ENV=production

# Use Gunicorn for production
gunicorn -w 4 -b 0.0.0.0:$PORT run:app
```

## API Documentation


### Error Handling
The application includes comprehensive error handling with user-friendly messages.


## License

This project is developed for the Nirmaan AI Intern Case Study.

## Support

For issues and questions, please refer to the project documentation or create an issue in the repository.

## Acknowledgments

- Based on the Nirmaan AI Communication Program requirements
- Uses Azure Cognitive Services for advanced AI capabilities
- Built with modern web technologies for optimal user experience

