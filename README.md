<div align="center">

# 📚 Intelligent Exam Question Analysis & Agentic Assessment Design

### *From Educational Analytics to Autonomous Assessment Design*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An AI-driven educational analytics system that analyzes exam questions and evolves into an agentic AI assessment design assistant.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Deployment](#-deployment) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

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

## 🎯 Project Overview

This project implements a **two-milestone educational analytics system** that transforms from classical ML-based question analysis to an autonomous agentic AI assessment designer.

### Milestone 1: ML-Based Exam Question Analytics (Current)
Applies classical machine learning and NLP techniques to analyze exam questions and student responses, predicting difficulty levels and identifying learning gaps.

### Milestone 2: Agentic AI Assessment Design Assistant (Planned)
Extends the system into an agent-based AI application using **LangGraph** that autonomously reasons about assessment quality, retrieves pedagogical best practices, and generates structured improvements.

### 🎓 Educational Context

The system addresses real-world challenges in educational assessment:
- **Difficulty Calibration**: Automatically classify questions by difficulty
- **Learning Gap Identification**: Detect areas where students struggle
- **Assessment Quality**: Evaluate question effectiveness using discrimination index
- **Pedagogical Alignment**: Ensure questions align with Bloom's taxonomy
- **Data-Driven Insights**: Provide actionable analytics for educators

---

## ✨ Key Features

### 🔍 Question Analysis
- **Difficulty Prediction**: Classify questions as Easy, Medium, or Hard
- **Confidence Scoring**: Probability distribution across difficulty levels
- **Multi-Subject Support**: Mathematics, Physics, Computer Science, Engineering
- **Cognitive Level Mapping**: Bloom's taxonomy integration
- **Readability Analysis**: Text complexity metrics

### 📊 Analytics Dashboard
- **Real-time Predictions**: Instant difficulty assessment
- **Visual Insights**: Interactive charts and probability distributions
- **Batch Processing**: Analyze multiple questions simultaneously
- **Export Functionality**: Download results as CSV
- **Performance Metrics**: Detailed model evaluation

### 🎨 User Interface
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

---

## 🛠️ Technology Stack

### Core Technologies

| Category | Technology | Purpose |
|----------|-----------|---------|
| **ML/NLP** | Scikit-learn 1.3+ | Machine learning algorithms |
| **Vectorization** | TF-IDF | Text feature extraction |
| **Models** | Logistic Regression, Decision Trees, Random Forest | Classification |
| **UI Framework** | Streamlit 1.28+ | Interactive web application |
| **Data Processing** | Pandas 2.0+, NumPy 1.24+ | Data manipulation |
| **Model Persistence** | Joblib 1.3+ | Model serialization |
| **Visualization** | Matplotlib, Plotly | Data visualization |

### Future Technologies (Milestone 2)

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Agent Framework** | LangGraph | Agentic workflow orchestration |
| **LLM** | Open-source models (Free tier) | Natural language understanding |
| **RAG** | Chroma/FAISS | Pedagogical knowledge retrieval |
| **State Management** | LangGraph State | Workflow state tracking |

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning repository)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/exam-question-analysis.git
cd exam-question-analysis
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

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import sklearn, streamlit, pandas; print('✅ All dependencies installed!')"
```

---

## 🚀 Quick Start

### 1️⃣ Train the Model

```bash
python train_model.py
```

**Expected Output:**
```
Loading dataset...
✅ Loaded dataset: 5000 questions

Dataset size: 5000 questions
Difficulty distribution:
hard      1703
easy      1652
medium    1645

Training TF-IDF Vectorizer...
Training models...

Best model: Logistic Regression with accuracy 0.314
✅ Models saved successfully!
```

### 2️⃣ Launch Application

```bash
streamlit run app.py
```

**Access the application at:** `http://localhost:8501`

### 3️⃣ Analyze Questions

1. Navigate to **"Question Analysis"** tab
2. Enter your exam question
3. Add optional metadata (subject, grade, type)
4. Click **"🔍 Analyze Question"**
5. View results and recommendations

---

## 📁 Project Structure

```
exam-question-analysis/
│
├── 📄 app.py                          # Streamlit web application
├── 📄 train_model.py                  # Model training pipeline
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Project documentation
├── 📄 .gitignore                      # Git ignore rules
├── 📄 setup.sh                        # Deployment setup script
│
├── 📂 models/                         # Trained models (generated)
│   ├── difficulty_model.pkl          # Trained classifier
│   └── tfidf_vectorizer.pkl          # TF-IDF vectorizer
│
├── 📂 data/                           # Dataset files
│   ├── question_ans_analysis.csv     # Main dataset (5000 questions)
│   └── sample_questions.csv          # Sample data for testing
│
├── 📂 .streamlit/                     # Streamlit configuration
│   └── config.toml                   # UI theme and settings
│
├── 📂 notebooks/                      # Jupyter notebooks (optional)
│   └── GENAI.ipynb                   # Exploratory data analysis
│
└── 📂 docs/                           # Additional documentation
    ├── API.md                        # API documentation
    ├── DEPLOYMENT.md                 # Deployment guide
    └── CONTRIBUTING.md               # Contribution guidelines
```

---

## 📊 Data Schema

### Input Dataset: `question_ans_analysis.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `question_text` | string | The exam question text | "Solve the quadratic equation..." |
| `subject` | string | Academic subject | Mathematics, Physics, CS |
| `cognitive_level_bloom` | string | Bloom's taxonomy level | remember, understand, apply, analyze, evaluate, create |
| `readability_score` | float | Text readability metric | 45.01 - 89.84 |
| `word_count` | int | Number of words | 15 - 49 |
| `sentence_count` | int | Number of sentences | 1 - 3 |
| `time_taken_minutes` | int | Average completion time | 3 - 24 minutes |
| `total_students_attempted` | int | Total attempts | 100 - 300 |
| `correct_attempts` | int | Successful attempts | Varies |
| `incorrect_attempts` | int | Failed attempts | Varies |
| `correct_percentage` | float | Success rate | 0.0 - 1.0 |
| `learning_gap_score` | float | Knowledge gap indicator | 0.0 - 1.0 |
| `discrimination_index` | float | Question quality metric | 0.0 - 1.0 |
| `difficulty_label` | string | **Target variable** | easy, medium, hard |
| `assessment_quality_score` | float | Overall quality | 0.0 - 1.0 |

### Dataset Statistics

- **Total Questions**: 5,000
- **Subjects**: 4 (Mathematics, Physics, Computer Science, Engineering Aptitude)
- **Difficulty Distribution**: 
  - Hard: 1,703 (34.1%)
  - Easy: 1,652 (33.0%)
  - Medium: 1,645 (32.9%)

---

## 🧠 Model Architecture

### Feature Engineering Pipeline

```python
Text Input → Preprocessing → TF-IDF Vectorization → ML Model → Prediction
```

#### 1. Text Preprocessing
- Lowercase conversion
- Special character removal
- Tokenization
- Stop word removal (optional)

#### 2. Feature Extraction
- **TF-IDF Vectorization**
  - Max features: 100
  - N-gram range: (1, 2)
  - Stop words: English

#### 3. Classification Models

| Model | Algorithm | Hyperparameters |
|-------|-----------|-----------------|
| **Logistic Regression** | Linear classifier | max_iter=1000, random_state=42 |
| **Decision Tree** | Tree-based classifier | random_state=42 |
| **Random Forest** | Ensemble method | n_estimators=100, random_state=42 |

#### 4. Model Selection
- Best model selected based on accuracy
- Cross-validation for robustness
- Stratified train-test split (80-20)

### Training Process

```bash
python train_model.py
```

**Pipeline Steps:**
1. Load dataset from CSV
2. Map difficulty labels to numeric (easy=0, medium=1, hard=2)
3. Split data (80% train, 20% test)
4. Train TF-IDF vectorizer on training data
5. Train multiple classifiers
6. Evaluate on test set
7. Select best model
8. Save model and vectorizer using joblib

---

## 📖 Usage Guide

### Single Question Analysis

#### Via Web Interface

1. **Launch Application**
   ```bash
   streamlit run app.py
   ```

2. **Navigate to "Question Analysis"**

3. **Enter Question Details**
   - Question text (required)
   - Subject (optional)
   - Grade level (optional)
   - Question type (optional)

4. **Analyze**
   - Click "🔍 Analyze Question"
   - View difficulty prediction
   - Check confidence scores
   - Read recommendations

#### Example

**Input:**
```
Question: "Derive the quadratic formula from the general quadratic equation."
Subject: Mathematics
Grade: 12
Type: Problem Solving
```

**Output:**
```
🔴 Hard
Confidence: 78.5%
Word Count: 10

Recommendations:
⚠️ This is a challenging question. Consider for advanced students or final exams.
```

### Batch Analysis

#### Prepare CSV File

Create a CSV file with a `question` column:

```csv
question
What is 2 + 2?
Explain the theory of relativity
Calculate the derivative of x^2
```

#### Upload and Analyze

1. Navigate to **"Batch Analysis"** tab
2. Upload CSV file
3. Click **"Analyze All Questions"**
4. Download results

#### Output Format

```csv
question,predicted_difficulty,difficulty_label,confidence_easy,confidence_medium,confidence_hard
What is 2 + 2?,0,Easy,0.85,0.10,0.05
Explain the theory of relativity,2,Hard,0.05,0.15,0.80
```

---

## 🔌 API Documentation

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
    
    difficulty_map = {0: "Easy", 1: "Medium", 2: "Hard"}
    return {
        'difficulty': difficulty_map[prediction],
        'confidence': max(probability),
        'probabilities': {
            'easy': probability[0],
            'medium': probability[1],
            'hard': probability[2]
        }
    }

# Example
result = predict_difficulty("What is the capital of France?")
print(result)
# Output: {'difficulty': 'Easy', 'confidence': 0.85, ...}
```

---

## 📈 Performance Metrics

### Model Evaluation (Test Set)

| Metric | Easy | Medium | Hard | Overall |
|--------|------|--------|------|---------|
| **Precision** | 0.31 | 0.31 | 0.32 | 0.31 |
| **Recall** | 0.29 | 0.23 | 0.42 | 0.31 |
| **F1-Score** | 0.30 | 0.26 | 0.36 | 0.31 |
| **Support** | 330 | 329 | 341 | 1000 |

### Confusion Matrix

```
              Predicted
              Easy  Med  Hard
Actual Easy    96   120   114
       Med     76    76   177
       Hard    98   101   142
```

### Key Insights

- ✅ Model performs best on Hard questions (42% recall)
- ⚠️ Medium questions are most challenging to classify
- 🎯 Overall accuracy: 31.4% (baseline: 33.3%)
- 📊 Balanced precision across all classes

---

## 🌐 Deployment

### Streamlit Community Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [streamlit.io/cloud](https://streamlit.io/cloud)
   - Connect GitHub repository
   - Select `app.py` as main file
   - Deploy!

### Hugging Face Spaces

1. **Create Space**
   - Visit [huggingface.co/spaces](https://huggingface.co/spaces)
   - Create new Streamlit Space

2. **Upload Files**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
   cp -r * YOUR_SPACE/
   cd YOUR_SPACE
   git add .
   git commit -m "Deploy app"
   git push
   ```

### Render (Free Tier)

1. **Create `render.yaml`**
   ```yaml
   services:
     - type: web
       name: exam-analysis
       env: python
       buildCommand: pip install -r requirements.txt && python train_model.py
       startCommand: streamlit run app.py
   ```

2. **Deploy**
   - Connect GitHub repository
   - Render auto-deploys on push

---

## 🗺️ Roadmap

### ✅ Milestone 1: ML-Based Analytics (Completed)
- [x] Data preprocessing pipeline
- [x] TF-IDF feature extraction
- [x] Multiple ML models training
- [x] Streamlit web interface
- [x] Batch processing support
- [x] Model persistence with joblib

### 🚧 Milestone 2: Agentic AI Assistant (In Progress)
- [ ] LangGraph workflow integration
- [ ] RAG system for pedagogical knowledge
- [ ] LLM integration (open-source)
- [ ] Autonomous reasoning engine
- [ ] Assessment improvement generator
- [ ] Structured output formatting

### 🔮 Future Enhancements
- [ ] Multi-language support
- [ ] Real-time collaboration
- [ ] Question bank management
- [ ] Student performance tracking
- [ ] Adaptive difficulty adjustment
- [ ] API endpoints for integration

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
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

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Educational assessment data from academic institutions
- **Frameworks**: Scikit-learn, Streamlit, Pandas communities
- **Inspiration**: Modern educational technology research
- **Contributors**: All project contributors and testers

---

## 📞 Contact & Support

- **Project Lead**: [Your Name]
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Issues**: [GitHub Issues](https://github.com/yourusername/exam-question-analysis/issues)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for Education

</div>
