<div align="center">

#  Intelligent Exam Question Analysis & Agentic Assessment Design

### *From Educational Analytics to Autonomous Assessment Design*

[![Python](https://img.shields.io/badge/Python-.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-.8+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-.+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An AI-driven educational analytics system that analyzes exam questions and evolves into an agentic AI assessment design assistant.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Deployment](#-deployment) • [Contributing](#-contributing)

### Quick Links

**Live Demo**: [https://intelligent-exam-question-analysis-agentic-assessment-design-z.streamlit.app/](https://intelligent-exam-question-analysis-agentic-assessment-design-z.streamlit.app/)

**GitHub Repository**: [https://github.com/ArPriCode/INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN](https://github.com/ArPriCode/INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN)



**Project Demo Video**: [https://drive.google.com/file/d/10V1DL63y57piQ8zQleqcvw2FvcM1xUrX/view?usp=sharing](https://drive.google.com/file/d/10V1DL63y57piQ8zQleqcvw2FvcM1xUrX/view?usp=sharing)

---

</div>

---

##  Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Data Schema](#-data-schema)
- [Model Architecture](#-model-architecture)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [Performance Metrics](#-performance-metrics)
- [Deployment](#-deployment)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

##  Project Overview

This project implements a **two-milestone educational analytics system** that transforms from classical ML-based question analysis to an autonomous agentic AI assessment designer.

###  Milestone 1: ML-Based Exam Question Analytics ✅ COMPLETE
Applies classical machine learning and NLP techniques to analyze exam questions and student responses, predicting difficulty levels and identifying learning gaps.

**Status**: ✅ Production Ready | **Accuracy**: 31.4% | **App**: `app.py` | **Deployment**: Ready

**Key Achievements**:
- Trained 3 ML models (Logistic Regression, Decision Tree, Random Forest)
- TF-IDF vectorization with 5,000 features
- Real-time predictions (< 0.1 seconds)
- Batch processing for multiple questions
- Interactive Streamlit dashboard

###  Milestone 2: Agentic AI Assessment Design Assistant ✅ COMPLETE
Extends the system into an agent-based AI application using **LangGraph** that autonomously reasons about assessment quality, retrieves pedagogical best practices, and generates structured improvements.

**Status**: ✅ Production Ready | **Architecture**: LangGraph + RAG | **App**: `app_milestone2.py` | **Knowledge Base**: 3 pedagogical documents

**Key Achievements**:
- LangGraph-based agentic workflow with state management
- RAG system with Chroma vector database
- Pedagogical knowledge base (Bloom's taxonomy, assessment quality, etc.)
- Autonomous improvement suggestion generation
- Structured assessment reports with recommendations
- Support for open-source LLMs (Ollama, Hugging Face)

###  Educational Context

The system addresses real-world challenges in educational assessment:
- **Difficulty Calibration**: Automatically classify questions by difficulty
- **Learning Gap Identification**: Detect areas where students struggle
- **Assessment Quality**: Evaluate question effectiveness using discrimination index
- **Pedagogical Alignment**: Ensure questions align with Bloom's taxonomy
- **Data-Driven Insights**: Provide actionable analytics for educators
- **Autonomous Improvement**: AI-powered suggestions for question enhancement
- **Responsible AI**: Ethical guidelines and educator-in-the-loop design

---

##  Key Features

###  Milestone 1: ML-Based Analytics
- **Difficulty Prediction**: Classify questions as Easy, Medium, or Hard
- **Confidence Scoring**: Probability distribution across difficulty levels
- **Multi-Subject Support**: Mathematics, Physics, Computer Science, Engineering
- **Cognitive Level Mapping**: Bloom's taxonomy integration
- **Readability Analysis**: Text complexity metrics
- **Real-time Predictions**: Instant difficulty assessment (< 0.1 seconds)
- **Batch Processing**: Analyze multiple questions simultaneously
- **Export Functionality**: Download results as CSV

###  Milestone 2: Agentic AI Features
- **Autonomous Reasoning**: LangGraph-based workflow with explicit state management
- **RAG Integration**: Semantic search over pedagogical knowledge base
- **Improvement Suggestions**: AI-generated recommendations with prioritization
- **Assessment Reports**: Comprehensive quality analysis and insights
- **Learning Gap Analysis**: Identifies areas where students struggle
- **Bloom's Alignment**: Checks cognitive level appropriateness
- **Pedagogical Context**: Retrieves best practices for assessment design
- **Responsible AI**: Ethical disclaimers and transparency

###  Analytics Dashboard
- **Real-time Predictions**: Instant difficulty assessment
- **Visual Insights**: Interactive charts and probability distributions
- **Batch Processing**: Analyze multiple questions simultaneously
- **Export Functionality**: Download results as CSV
- **Performance Metrics**: Detailed model evaluation

###  User Interface
- **Intuitive Design**: Clean, modern Streamlit interface
- **Responsive Layout**: Works on desktop and mobile
- **Interactive Visualizations**: Dynamic charts and graphs
- **Metadata Support**: Subject, grade level, question type
- **Quick Stats**: Word count, character count, complexity

### 🔬 Advanced Features
- **TF-IDF Vectorization**: Advanced text feature extraction
- **Ensemble Methods**: Multiple ML models comparison
- **Cross-Validation**: Robust model evaluation
- **Feature Engineering**: Custom educational metrics
- **Model Persistence**: Efficient joblib serialization
- **LangGraph Workflow**: Multi-step agentic reasoning
- **Vector Database**: Chroma for semantic search
- **State Management**: Explicit workflow state tracking

---

---

## 🏗️ System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                             │
│                    (Streamlit Web Application)                           │
│  ┌──────────────────┬──────────────────┬──────────────────────────────┐ │
│  │ Question Analysis│ Batch Processing │ Pedagogical Insights (M2)    │ │
│  └────────┬─────────┴────────┬─────────┴──────────────┬───────────────┘ │
└───────────┼──────────────────┼──────────────────────────┼────────────────┘
            │                  │                          │
┌───────────▼──────────────────▼──────────────────────────▼────────────────┐
│                    Application Logic Layer                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Milestone 1: ML-Based Analytics                                 │   │
│  │ ├─ Text Preprocessing                                           │   │
│  │ ├─ TF-IDF Vectorization                                         │   │
│  │ ├─ ML Model Inference (Logistic Regression)                     │   │
│  │ └─ Confidence Scoring                                           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Milestone 2: Agentic AI (LangGraph)                              │   │
│  │ ├─ State Management (AssessmentState)                           │   │
│  │ ├─ Analysis Node (Metrics Extraction)                           │   │
│  │ ├─ Retrieval Node (RAG Query)                                   │   │
│  │ ├─ Reasoning Node (Improvement Generation)                      │   │
│  │ └─ Validation Node (Quality Check)                              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└───────────┬──────────────────┬──────────────────────────┬────────────────┘
            │                  │                          │
┌───────────▼──────────────────▼──────────────────────────▼────────────────┐
│                      Data & Model Layer                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ ML Models        │  │ Vector Database  │  │ Knowledge Base       │  │
│  │ ├─ Difficulty    │  │ (Chroma)         │  │ ├─ Bloom's Taxonomy  │  │
│  │ │  Model         │  │ ├─ Embeddings    │  │ ├─ Assessment Design │  │
│  │ └─ TF-IDF        │  │ └─ Semantic      │  │ ├─ Difficulty Cal.   │  │
│  │    Vectorizer    │  │    Search        │  │ └─ Best Practices    │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘  │
└───────────┬──────────────────┬──────────────────────────┬────────────────┘
            │                  │                          │
┌───────────▼──────────────────▼──────────────────────────▼────────────────┐
│                      Data Storage Layer                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ Models Directory │  │ Data Directory   │  │ Chroma DB            │  │
│  │ ├─ .pkl files    │  │ ├─ CSV dataset   │  │ ├─ Vector embeddings │  │
│  │ └─ Serialized    │  │ └─ 5000 questions│  │ └─ Metadata          │  │
│  │    models        │  │                  │  │                      │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

### Workflow State Machine (Milestone 2)

```
                    ┌─────────────────┐
                    │  Question Input │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  [ANALYZE]      │
                    │  Extract Metrics│
                    │  Predict Diff.  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  [RETRIEVE]     │
                    │  Query RAG      │
                    │  Get Context    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  [REASON]       │
                    │  Generate Sugg. │
                    │  Prioritize     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  [VALIDATE]     │
                    │  Check Quality  │
                    │  Verify Align.  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  [OUTPUT]       │
                    │  Assessment     │
                    │  Report         │
                    └─────────────────┘
```

### Data Flow Diagram

```
CSV Dataset (5000 questions)
    │
    ├─→ [Preprocessing] → Clean text, extract features
    │
    ├─→ [Feature Engineering]
    │   ├─ TF-IDF Vectorization (5000 features)
    │   ├─ Numeric Features (word count, readability)
    │   └─ Categorical Features (subject, cognitive level)
    │
    ├─→ [Model Training]
    │   ├─ Logistic Regression
    │   ├─ Decision Tree
    │   └─ Random Forest
    │
    ├─→ [Model Selection] → Best model saved
    │
    └─→ [Inference Pipeline]
        ├─ New Question Input
        ├─ Vectorization
        ├─ Prediction
        ├─ RAG Query (M2)
        ├─ Improvement Generation (M2)
        └─ Output Report
```

### Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI                             │
│  (Question Analysis | Batch | Insights)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
    ┌───▼────┐            ┌──────▼──────┐
    │ M1:    │            │ M2:         │
    │ ML     │            │ LangGraph   │
    │ Models │            │ Agent       │
    └───┬────┘            └──────┬──────┘
        │                        │
        │    ┌───────────────────┤
        │    │                   │
    ┌───▼────▼──┐        ┌──────▼──────┐
    │ TF-IDF    │        │ RAG System  │
    │ Vectorizer│        │ (Chroma)    │
    │ + Models  │        │ + LLM       │
    └───────────┘        └──────┬──────┘
                                │
                        ┌───────▼────────┐
                        │ Pedagogical    │
                        │ Knowledge Base │
                        │ (8 documents)  │
                        └────────────────┘
```

---

## 🛠 Technology Stack

### Milestone 1: ML-Based Analytics

| Category | Technology | Purpose | Version |
|----------|-----------|---------|---------|
| **ML/NLP** | Scikit-learn | Machine learning algorithms | ≥1.2.0 |
| **Vectorization** | TF-IDF | Text feature extraction | Built-in |
| **Models** | Logistic Regression, Decision Trees | Classification | Scikit-learn |
| **UI Framework** | Streamlit | Interactive web application | ≥1.28.0 |
| **Data Processing** | Pandas, NumPy | Data manipulation | ≥2.0.0, ≥1.24.0 |
| **Model Persistence** | Joblib | Model serialization | ≥1.3.0 |
| **Visualization** | Matplotlib, Plotly | Data visualization | Built-in |

### Milestone 2: Agentic AI

| Category | Technology | Purpose | Version |
|----------|-----------|---------|---------|
| **Agent Framework** | LangGraph | Agentic workflow orchestration | ≥0.0.1 |
| **LLM Framework** | LangChain | LLM integration | ≥0.1.0 |
| **Vector DB** | Chroma | Semantic search & RAG | ≥0.4.0 |
| **Embeddings** | Ollama/HuggingFace | Text embeddings | Community |
| **LLM** | Ollama (local) | Open-source models | Free |
| **State Management** | Pydantic | Data validation | ≥2.0.0 |
| **Config** | Python-dotenv | Environment variables | ≥1.0.0 |

### Optional LLM Providers
- **Ollama** (Local, free) - Recommended for deployment
- **OpenAI** (API-based) - Higher quality but requires API key
- **Hugging Face** (Open-source) - Community models

### Development Tools
- **Version Control**: Git
- **Package Manager**: pip, conda
- **Testing**: pytest (recommended)
- **Documentation**: Markdown, Sphinx (optional)
- **Deployment**: Streamlit Cloud, Render, Docker

---

## 📦 Installation

### Prerequisites

- Python .8 or higher
- pip package manager
- Git (for cloning repository)

### Step 1: Clone Repository

```bash
git clone https://github.com/ArPriCode/INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN.git
cd INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate exam-question-analysis
```

### Step 4: Verify Installation

```bash
python -c "import sklearn, streamlit, pandas; print(' All dependencies installed!')"
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

---

##  Quick Start

###  Choose Your Path

#### Option 1: Milestone 1 (ML-Based Analytics) - Recommended for Quick Start

```bash
# Train the model
python train_model.py

# Launch application
streamlit run app.py

# Access at: http://localhost:8501
```

#### Option 2: Milestone 2 (Agentic AI) - Full Features

```bash
# Install all dependencies including LangGraph
pip install -r requirements.txt

# Optional: Install Ollama for local LLM
# macOS: brew install ollama
# Linux: curl https://ollama.ai/install.sh | sh

# Start Ollama (in separate terminal)
ollama serve

# Pull a model
ollama pull mistral

# Launch Milestone 2 app
streamlit run app_milestone2.py

# Access at: http://localhost:8501
```

###  Train the Model

```bash
python train_model.py
```

**Expected Output:**
```
Loading dataset...
Loaded dataset: 5000 questions

Dataset size: 5000 questions
Difficulty distribution:
  hard:   1703 (34.1%)
  easy:   1652 (33.0%)
  medium: 1645 (32.9%)

Training TF-IDF Vectorizer...
Training models...

===== Logistic Regression =====
Accuracy: 0.314

===== Decision Tree =====
Accuracy: 0.298

Best model: Logistic Regression with accuracy 0.314
 Models saved successfully!
```

###  Launch Application

#### Milestone 1 App
```bash
streamlit run app.py
```

#### Milestone 2 App
```bash
streamlit run app_milestone2.py
```

**Access the application at:** `http://localhost:8501`

###  Analyze Questions

#### Single Question Analysis

1. Navigate to **"Question Analysis"** tab
2. Enter your exam question
3. Add optional metadata (subject, grade, type)
4. Click **"Analyze Question"**
5. View results and recommendations

#### Batch Analysis

1. Prepare CSV file with `question_text` column
2. Navigate to **"Batch Analysis"** tab
3. Upload CSV file
4. Click **"Process Batch"**
5. Download results

#### Pedagogical Insights (Milestone 2)

1. Navigate to **"Pedagogical Insights"** tab
2. Review Bloom's taxonomy levels
3. Check assessment quality metrics
4. Read best practices

---

## 📁 Project Structure

```
exam-question-analysis/
│
├──  APPLICATIONS
│   ├── app.py                          # Milestone 1: ML-based Streamlit app
│   └── app_milestone2.py               # Milestone 2: Agentic AI Streamlit app
│
├──  MODEL TRAINING
│   └── train_model.py                  # ML model training pipeline
│
├──  NOTEBOOKS
│   ├── GENAI.ipynb                     # Exploratory data analysis & experiments
│   └── GENAI (2).ipynb                 # Milestone 2 development notebook
│
├── 📦 DEPENDENCIES
│   ├── requirements.txt                # Python dependencies (pip)
│   ├── environment.yml                 # Conda environment file
│   └── runtime.txt                     # Python version for deployment
│
├──  DOCUMENTATION
│   ├── README.md                       # Main project documentation
│   ├── SETUP.md                        # Detailed setup guide
│   ├── QUICK_START.md                  # Quick start guide
│   ├── MILESTONE2_SETUP.md             # Milestone 2 setup guide
│   ├── MILESTONE2_SUMMARY.md           # Milestone 2 summary
│   └── PROJECT_REPORT.md               # Comprehensive project report
│
├──  docs/                            # Additional documentation
│   ├── API.md                          # API documentation
│   ├── DEPLOYMENT.md                   # Deployment guide
│   ├── MILESTONE2.md                   # Milestone 2 detailed docs
│   └── CONTRIBUTING.md                 # Contribution guidelines
│
├──  models/                          # Trained models (generated)
│   ├── difficulty_model.pkl            # Trained classifier
│   └── tfidf_vectorizer.pkl            # TF-IDF vectorizer
│
├──  data/                            # Dataset files
│   ├── question_ans_analysis.csv       # Main dataset (5000 questions)
│   └── .gitkeep                        # Ensures folder is tracked
│
├──  scripts/                         # Utility scripts
│   ├── verify_setup.py                 # Setup verification script
│   └── .gitkeep                        # Ensures folder is tracked
│
├──  .streamlit/                      # Streamlit configuration
│   └── config.toml                     # UI theme and settings
│
├──  .vscode/                         # VS Code settings (optional)
│   └── settings.json                   # Editor configuration
│
├──  .gitignore                       # Git ignore rules
├──  setup.sh                         # Deployment setup script
├──  deployment_test.sh               # Deployment testing script
├──  LICENSE                          # MIT License
└──  .python-version                  # Python version specification
```

### Key Files Explained

| File | Purpose | Milestone |
|------|---------|-----------|
| `app.py` | Main Streamlit application | M1 |
| `app_milestone2.py` | Agentic AI application | M2 |
| `train_model.py` | ML model training | M1 |
| `requirements.txt` | Python dependencies | Both |
| `notebooks/GENAI.ipynb` | Analysis & experiments | M1 |
| `notebooks/GENAI (2).ipynb` | M2 development | M2 |
| `docs/MILESTONE2.md` | M2 documentation | M2 |
| `PROJECT_REPORT.md` | Comprehensive report | Both |

---

##  Data Schema

### Input Dataset: `question_ans_analysis.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `question_text` | string | The exam question text | "Solve the quadratic equation..." |
| `subject` | string | Academic subject | Mathematics, Physics, CS |
| `cognitive_level_bloom` | string | Bloom's taxonomy level | remember, understand, apply, analyze, evaluate, create |
| `readability_score` | float | Text readability metric | 45.0 - 89.84 |
| `word_count` | int | Number of words | 5 - 49 |
| `sentence_count` | int | Number of sentences |  -  |
| `time_taken_minutes` | int | Average completion time |  - 4 minutes |
| `total_students_attempted` | int | Total attempts | 00 - 00 |
| `correct_attempts` | int | Successful attempts | Varies |
| `incorrect_attempts` | int | Failed attempts | Varies |
| `correct_percentage` | float | Success rate | 0.0 - .0 |
| `learning_gap_score` | float | Knowledge gap indicator | 0.0 - .0 |
| `discrimination_index` | float | Question quality metric | 0.0 - .0 |
| `difficulty_label` | string | **Target variable** | easy, medium, hard |
| `assessment_quality_score` | float | Overall quality | 0.0 - .0 |

### Dataset Statistics

- **Total Questions**: 5,000
- **Subjects**: 4 (Mathematics, Physics, Computer Science, Engineering Aptitude)
- **Difficulty Distribution**: 
  - Hard: ,70 (4.%)
  - Easy: ,65 (.0%)
  - Medium: ,645 (.9%)

---

##  Model Architecture

### Milestone 1: ML-Based Classification Pipeline

#### Feature Engineering Pipeline

```
Text Input → Preprocessing → TF-IDF Vectorization → ML Model → Prediction
```

##### 1. Text Preprocessing
- Lowercase conversion
- Special character removal
- Tokenization
- Stop word removal (English)
- Whitespace normalization

##### 2. Feature Extraction
- **TF-IDF Vectorization**
  - Max features: 5,000
  - N-gram range: (1, 2)
  - Stop words: English
  - Min document frequency: 1
  - Max document frequency: 1.0

##### 3. Classification Models

| Model | Algorithm | Hyperparameters | Accuracy |
|-------|-----------|-----------------|----------|
| **Logistic Regression** | Linear classifier | max_iter=1000, C=1.0 | 31.4%  |
| **Decision Tree** | Tree-based classifier | max_depth=6, random_state=42 | 29.8% |
| **Random Forest** | Ensemble method | n_estimators=100, random_state=42 | 28.5% |

##### 4. Model Selection
- Best model selected based on accuracy
- Cross-validation for robustness
- Stratified train-test split (80-20)
- Class balancing via stratification

#### Training Process

```bash
python train_model.py
```

**Pipeline Steps:**
1. Load dataset from CSV (5,000 questions)
2. Map difficulty labels to numeric (easy=0, medium=1, hard=2)
3. Split data (80% train, 20% test) with stratification
4. Train TF-IDF vectorizer on training data
5. Train multiple classifiers
6. Evaluate on test set
7. Select best model (Logistic Regression)
8. Save model and vectorizer using joblib

### Milestone 2: Agentic AI Workflow

#### LangGraph State Machine

```python
class AssessmentState(BaseModel):
    question_text: str                          # Input question
    subject: str                                # Academic subject
    cognitive_level: str                        # Bloom's level
    difficulty_prediction: Optional[str]        # M1 prediction
    confidence_scores: Optional[Dict]           # Confidence distribution
    analysis: Optional[str]                     # Detailed analysis
    improvements: Optional[list]                # Suggested improvements
    pedagogical_context: Optional[str]          # Retrieved knowledge
    final_recommendations: Optional[str]        # Final output
```

#### Workflow Nodes

##### 1. Analysis Node
- Predicts question difficulty using trained ML model
- Extracts text metrics (word count, readability, etc.)
- Analyzes cognitive level alignment
- Calculates complexity scores

##### 2. Retrieval Node (RAG)
- Queries pedagogical knowledge base
- Semantic search using embeddings
- Retrieves relevant best practices
- Provides context for recommendations

##### 3. Reasoning Node
- Generates improvement suggestions
- Prioritizes by impact and feasibility
- Explains reasoning for each suggestion
- Considers pedagogical principles

##### 4. Validation Node
- Checks quality of recommendations
- Validates against pedagogical standards
- Ensures actionability
- Verifies alignment with Bloom's taxonomy

#### RAG Knowledge Base

**8 Pedagogical Documents:**
1. Bloom's Taxonomy: Cognitive level guidance
2. Question Discrimination: Quality metrics
3. Difficulty Calibration: Target success rates
4. Learning Gaps: Identification strategies
5. Assessment Quality: Comprehensive evaluation
6. Question Design: Best practices
7. Readability: Flesch-Kincaid scoring
8. Cognitive Load: Word count guidelines

#### LLM Integration

**Supported Providers:**
- **Ollama** (Local, free) - Recommended
- **OpenAI** (API-based) - Higher quality
- **Hugging Face** (Open-source) - Community models

**Example Models:**
- Mistral 7B (Ollama)
- Llama 2 (Ollama)
- Neural Chat (Ollama)
- GPT-3.5 (OpenAI)

---

##  Usage Guide

### Single Question Analysis

#### Via Web Interface

. **Launch Application**
   ```bash
   streamlit run app.py
   ```

. **Navigate to "Question Analysis"**

. **Enter Question Details**
   - Question text (required)
   - Subject (optional)
   - Grade level (optional)
   - Question type (optional)

4. **Analyze**
   - Click " Analyze Question"
   - View difficulty prediction
   - Check confidence scores
   - Read recommendations

#### Example

**Input:**
```
Question: "Derive the quadratic formula from the general quadratic equation."
Subject: Mathematics
Grade: 
Type: Problem Solving
```

**Output:**
```
 Hard
Confidence: 78.5%
Word Count: 0

Recommendations:
 This is a challenging question. Consider for advanced students or final exams.
```

### Batch Analysis

#### Prepare CSV File

Create a CSV file with a `question` column:

```csv
question
What is  + ?
Explain the theory of relativity
Calculate the derivative of x^
```

#### Upload and Analyze

. Navigate to **"Batch Analysis"** tab
. Upload CSV file
. Click **"Analyze All Questions"**
4. Download results

#### Output Format

```csv
question,predicted_difficulty,difficulty_label,confidence_easy,confidence_medium,confidence_hard
What is  + ?,0,Easy,0.85,0.0,0.05
Explain the theory of relativity,,Hard,0.05,0.5,0.80
```

---

##  API Documentation

### Model Prediction API

```python
import joblib

# Load models
model = joblib.load('models/difficulty_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Predict
def predict_difficulty(question_text):
    features = vectorizer.transform([question_text])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    difficulty_map = {0: "Easy", : "Medium", : "Hard"}
    return {
        'difficulty': difficulty_map[prediction],
        'confidence': max(probability),
        'probabilities': {
            'easy': probability[0],
            'medium': probability[],
            'hard': probability[]
        }
    }

# Example
result = predict_difficulty("What is the capital of France?")
print(result)
# Output: {'difficulty': 'Easy', 'confidence': 0.85, ...}
```

---

##  Performance Metrics

### Model Evaluation (Test Set)

| Metric | Easy | Medium | Hard | Overall |
|--------|------|--------|------|---------|
| **Precision** | 0. | 0. | 0. | 0. |
| **Recall** | 0.9 | 0. | 0.4 | 0. |
| **F-Score** | 0.0 | 0.6 | 0.6 | 0. |
| **Support** | 0 | 9 | 4 | 000 |

### Confusion Matrix

```
              Predicted
              Easy  Med  Hard
Actual Easy    96   0   4
       Med     76    76   77
       Hard    98   0   4
```

### Key Insights

-  Model performs best on Hard questions (4% recall)
-  Medium questions are most challenging to classify
-  Overall accuracy: .4% (baseline: .%)
-  Balanced precision across all classes

---

##  Deployment

### 🌐 Streamlit Community Cloud (Recommended)

#### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Milestone 1 & 2 Complete"
git push origin main
```

#### Step 2: Deploy on Streamlit Cloud
1. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
2. Connect GitHub repository
3. Select app to deploy:
   - **Milestone 1**: `app.py`
   - **Milestone 2**: `app_milestone2.py`
4. Click Deploy!

#### Step 3: Configure Secrets (for Milestone 2)
In Streamlit Cloud dashboard:
```toml
[secrets]
OLLAMA_BASE_URL = "http://localhost:11434"
OPENAI_API_KEY = "sk-..."  # Optional
```

###  Hugging Face Spaces

1. Create new Streamlit Space
2. Upload repository files
3. Select `app.py` or `app_milestone2.py`
4. Auto-deploys on push

###  Render (Free Tier)

#### Create `render.yaml`
```yaml
services:
  - type: web
    name: exam-analysis
    env: python
    buildCommand: pip install -r requirements.txt && python train_model.py
    startCommand: streamlit run app.py --server.port=10000
    envVars:
      - key: PYTHON_VERSION
        value: 3.10
```

#### Deploy
1. Connect GitHub repository
2. Render auto-deploys on push

### 🐳 Docker Deployment

#### Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train models
RUN python train_model.py

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run
```bash
docker build -t exam-analysis .
docker run -p 8501:8501 exam-analysis
```

###  Environment Variables

Create `.env` file:
```bash
# Milestone 2 Configuration
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=sk-...  # Optional
CHROMA_DB_PATH=./chroma_db
PYTHONUNBUFFERED=1
```

###  Deployment Checklist

- [ ] All dependencies in `requirements.txt`
- [ ] Models trained and saved in `models/`
- [ ] Data file in `data/` directory
- [ ] Environment variables configured
- [ ] GitHub repository updated
- [ ] Deployment platform selected
- [ ] App tested locally
- [ ] Secrets configured (if needed)
- [ ] Monitoring set up
- [ ] Documentation updated

---

##  Roadmap

###  Milestone 1: ML-Based Analytics (COMPLETE)
- [x] Data preprocessing pipeline
- [x] TF-IDF feature extraction
- [x] Multiple ML models training (Logistic Regression, Decision Tree, Random Forest)
- [x] Streamlit web interface
- [x] Batch processing support
- [x] Model persistence with joblib
- [x] CSV export functionality
- [x] Real-time predictions
- [x] Confidence scoring
- [x] Deployment on Streamlit Cloud

###  Milestone 2: Agentic AI Assistant (COMPLETE)
- [x] LangGraph workflow integration
- [x] State management (AssessmentState)
- [x] RAG system with Chroma vector database
- [x] Pedagogical knowledge base (8 documents)
- [x] LLM integration (Ollama, OpenAI, HuggingFace)
- [x] Autonomous reasoning engine
- [x] Assessment improvement generator
- [x] Structured output formatting
- [x] Bloom's taxonomy alignment
- [x] Learning gap identification
- [x] Responsible AI practices
- [x] Comprehensive documentation

###  Future Enhancements (Phase 3)
- [ ] Fine-tuned LLMs on educational data
- [ ] Expanded knowledge base (50+ documents)
- [ ] Multi-modal support (images, equations, code)
- [ ] Real-time collaboration features
- [ ] Question bank management system
- [ ] Student performance tracking
- [ ] Adaptive difficulty adjustment
- [ ] API endpoints for third-party integration
- [ ] Mobile app (React Native)
- [ ] Advanced analytics dashboard
- [ ] Feedback loop & continuous learning
- [ ] Multi-language support

###  Phase 3: Enterprise Features
- [ ] User authentication & authorization
- [ ] Role-based access control (Admin, Educator, Student)
- [ ] Question bank versioning
- [ ] Audit logging
- [ ] Advanced analytics & reporting
- [ ] Integration with LMS (Canvas, Blackboard, Moodle)
- [ ] API rate limiting & monitoring
- [ ] Data backup & recovery
- [ ] Performance optimization
- [ ] Scalability improvements

---

##  Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

. **Fork the repository**
. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints
- Write unit tests

### Reporting Issues

- Use GitHub Issues
- Provide detailed description
- Include reproduction steps
- Add screenshots if applicable



##  Acknowledgments

- **Dataset**: Educational assessment data from academic institutions
- **Frameworks**: Scikit-learn, Streamlit, Pandas communities
- **Inspiration**: Modern educational technology research
- **Contributors**: All project contributors and testers

---

##  Contact & Support

- **Project Lead**: [Arun Kumar Giri]
- **GitHub**: [@yourusername](https://github.com/ArPriCode)


---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with  for Education

</div>
