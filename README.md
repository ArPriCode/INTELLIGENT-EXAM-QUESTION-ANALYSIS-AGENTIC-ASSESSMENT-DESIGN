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

**Project Documentation**: [https://docs.google.com/document/d/1LRctuu2P9yFw6RQKCJO0vOb89xWeLaMxcevKJkV_lJA/edit?tab=t.0](https://docs.google.com/document/d/1LRctuu2P9yFw6RQKCJO0vOb89xWeLaMxcevKJkV_lJA/edit?tab=t.0)

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

### Milestone : ML-Based Exam Question Analytics (Current)
Applies classical machine learning and NLP techniques to analyze exam questions and student responses, predicting difficulty levels and identifying learning gaps.

### Milestone : Agentic AI Assessment Design Assistant (Planned)
Extends the system into an agent-based AI application using **LangGraph** that autonomously reasons about assessment quality, retrieves pedagogical best practices, and generates structured improvements.

###  Educational Context

The system addresses real-world challenges in educational assessment:
- **Difficulty Calibration**: Automatically classify questions by difficulty
- **Learning Gap Identification**: Detect areas where students struggle
- **Assessment Quality**: Evaluate question effectiveness using discrimination index
- **Pedagogical Alignment**: Ensure questions align with Bloom's taxonomy
- **Data-Driven Insights**: Provide actionable analytics for educators

---

##  Key Features

###  Question Analysis
- **Difficulty Prediction**: Classify questions as Easy, Medium, or Hard
- **Confidence Scoring**: Probability distribution across difficulty levels
- **Multi-Subject Support**: Mathematics, Physics, Computer Science, Engineering
- **Cognitive Level Mapping**: Bloom's taxonomy integration
- **Readability Analysis**: Text complexity metrics

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

---

## 🛠 Technology Stack

### Core Technologies

| Category | Technology | Purpose |
|----------|-----------|---------|
| **ML/NLP** | Scikit-learn .+ | Machine learning algorithms |
| **Vectorization** | TF-IDF | Text feature extraction |
| **Models** | Logistic Regression, Decision Trees, Random Forest | Classification |
| **UI Framework** | Streamlit .8+ | Interactive web application |
| **Data Processing** | Pandas .0+, NumPy .4+ | Data manipulation |
| **Model Persistence** | Joblib .+ | Model serialization |
| **Visualization** | Matplotlib, Plotly | Data visualization |

### Future Technologies (Milestone )

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Agent Framework** | LangGraph | Agentic workflow orchestration |
| **LLM** | Open-source models (Free tier) | Natural language understanding |
| **RAG** | Chroma/FAISS | Pedagogical knowledge retrieval |
| **State Management** | LangGraph State | Workflow state tracking |

---

## 📦 Installation

### Prerequisites

- Python .8 or higher
- pip package manager
- Git (for cloning repository)

### Step : Clone Repository

```bash
git clone https://github.com/yourusername/exam-question-analysis.git
cd exam-question-analysis
```

### Step : Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step : Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import sklearn, streamlit, pandas; print(' All dependencies installed!')"
```

---

##  Quick Start

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
hard      70
easy      65
medium    645

Training TF-IDF Vectorizer...
Training models...

Best model: Logistic Regression with accuracy 0.4
 Models saved successfully!
```

###  Launch Application

```bash
streamlit run app.py
```

**Access the application at:** `http://localhost:850`

###  Analyze Questions

. Navigate to **"Question Analysis"** tab
. Enter your exam question
. Add optional metadata (subject, grade, type)
4. Click **" Analyze Question"**
5. View results and recommendations

---

## 📁 Project Structure

```
exam-question-analysis/
│
├──  app.py                          # Streamlit web application
├──  train_model.py                  # Model training pipeline
├──  requirements.txt                # Python dependencies
├──  README.md                       # Project documentation
├──  .gitignore                      # Git ignore rules
├──  setup.sh                        # Deployment setup script
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

## 🧠 Model Architecture

### Feature Engineering Pipeline

```python
Text Input → Preprocessing → TF-IDF Vectorization → ML Model → Prediction
```

#### . Text Preprocessing
- Lowercase conversion
- Special character removal
- Tokenization
- Stop word removal (optional)

#### . Feature Extraction
- **TF-IDF Vectorization**
  - Max features: 00
  - N-gram range: (, )
  - Stop words: English

#### . Classification Models

| Model | Algorithm | Hyperparameters |
|-------|-----------|-----------------|
| **Logistic Regression** | Linear classifier | max_iter=000, random_state=4 |
| **Decision Tree** | Tree-based classifier | random_state=4 |
| **Random Forest** | Ensemble method | n_estimators=00, random_state=4 |

#### 4. Model Selection
- Best model selected based on accuracy
- Cross-validation for robustness
- Stratified train-test split (80-0)

### Training Process

```bash
python train_model.py
```

**Pipeline Steps:**
. Load dataset from CSV
. Map difficulty labels to numeric (easy=0, medium=, hard=)
. Split data (80% train, 0% test)
4. Train TF-IDF vectorizer on training data
5. Train multiple classifiers
6. Evaluate on test set
7. Select best model
8. Save model and vectorizer using joblib

---

## 📖 Usage Guide

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
🔴 Hard
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

## 📈 Performance Metrics

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

### Streamlit Community Cloud

. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

. **Deploy on Streamlit Cloud**
   - Visit [streamlit.io/cloud](https://streamlit.io/cloud)
   - Connect GitHub repository
   - Select `app.py` as main file
   - Deploy!

### Hugging Face Spaces

. **Create Space**
   - Visit [huggingface.co/spaces](https://huggingface.co/spaces)
   - Create new Streamlit Space

. **Upload Files**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
   cp -r * YOUR_SPACE/
   cd YOUR_SPACE
   git add .
   git commit -m "Deploy app"
   git push
   ```

### Render (Free Tier)

. **Create `render.yaml`**
   ```yaml
   services:
     - type: web
       name: exam-analysis
       env: python
       buildCommand: pip install -r requirements.txt && python train_model.py
       startCommand: streamlit run app.py
   ```

. **Deploy**
   - Connect GitHub repository
   - Render auto-deploys on push

---

##  Roadmap

###  Milestone : ML-Based Analytics (Completed)
- [x] Data preprocessing pipeline
- [x] TF-IDF feature extraction
- [x] Multiple ML models training
- [x] Streamlit web interface
- [x] Batch processing support
- [x] Model persistence with joblib

### 🚧 Milestone : Agentic AI Assistant (In Progress)
- [ ] LangGraph workflow integration
- [ ] RAG system for pedagogical knowledge
- [ ] LLM integration (open-source)
- [ ] Autonomous reasoning engine
- [ ] Assessment improvement generator
- [ ] Structured output formatting

###  Future Enhancements
- [ ] Multi-language support
- [ ] Real-time collaboration
- [ ] Question bank management
- [ ] Student performance tracking
- [ ] Adaptive difficulty adjustment
- [ ] API endpoints for integration

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

---

##  License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- **Dataset**: Educational assessment data from academic institutions
- **Frameworks**: Scikit-learn, Streamlit, Pandas communities
- **Inspiration**: Modern educational technology research
- **Contributors**: All project contributors and testers

---

##  Contact & Support

- **Project Lead**: [Your Name]
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Issues**: [GitHub Issues](https://github.com/yourusername/exam-question-analysis/issues)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with  for Education

</div>
