"""
Unified Application: Intelligent Exam Question Analysis & Agentic Assessment Design
Combines Milestone 1 (ML-Based Analytics) + Milestone 2 (Agentic AI)
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from typing import Optional, Dict, Any
import json

# Page config
st.set_page_config(
    page_title="Assessment Intelligence Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .improvement-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .milestone-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    .m1-badge {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .m2-badge {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        if not os.path.exists('models'):
            st.error("Models directory not found. Please run train_model.py first.")
            return None, None
        
        model = joblib.load('models/difficulty_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Pedagogical knowledge base for M2
PEDAGOGICAL_KNOWLEDGE = {
    "Bloom's Taxonomy": "Bloom's Taxonomy has 6 levels: Remember, Understand, Apply, Analyze, Evaluate, Create. Higher levels require deeper cognitive engagement and critical thinking.",
    "Discrimination Index": "Measures how well a question differentiates between high and low performers. Higher is better (0.3+). Questions with low discrimination don't effectively measure learning.",
    "Difficulty Calibration": "Easy questions (70%+ correct), Medium (45-65%), Hard (15-35%). Balance is key for effective assessment. Mix difficulty levels to maintain student engagement.",
    "Learning Gap Identification": "When many students fail a question, it indicates a knowledge gap. Provide remedial content and re-teach the concept before moving forward.",
    "Assessment Quality": "Combine difficulty, discrimination, and alignment with learning objectives for comprehensive evaluation. Quality questions are clear, fair, and aligned.",
    "Question Design Best Practices": "Use clear language, avoid ambiguity, align with learning objectives, test one concept per question. Avoid trick questions and cultural bias.",
    "Readability": "Flesch-Kincaid score 40-60 for easy, 60-80 for medium, 80+ for hard. Adjust language complexity accordingly. Shorter sentences improve clarity.",
    "Cognitive Load": "Shorter questions (15-25 words) reduce cognitive load. Longer questions (35+ words) increase difficulty. Balance information density.",
}

def predict_difficulty(question_text: str, model, vectorizer) -> tuple:
    """Milestone 1: Predict question difficulty"""
    try:
        X = vectorizer.transform([question_text])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Map numeric prediction to string
        difficulty_map = {0: 'Easy', 1: 'Medium', 2: 'Hard'}
        difficulty_str = difficulty_map.get(int(prediction), 'Unknown')
        
        confidence_scores = {
            'Easy': float(probabilities[0]),
            'Medium': float(probabilities[1]),
            'Hard': float(probabilities[2])
        }
        
        return difficulty_str, confidence_scores
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

def analyze_question_metrics(question_text: str) -> Dict[str, Any]:
    """Analyze question text metrics"""
    words = question_text.split()
    sentences = question_text.split('.')
    
    return {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'readability_estimate': 'Easy' if len(words) < 25 else ('Medium' if len(words) < 35 else 'Hard')
    }

def generate_improvements(question_text: str, difficulty: str, cognitive_level: str) -> list:
    """Milestone 2: Generate improvement suggestions"""
    improvements = []
    metrics = analyze_question_metrics(question_text)
    
    # Difficulty-based improvements
    if difficulty == 'Easy':
        improvements.append({
            'category': 'Difficulty',
            'suggestion': 'Consider increasing cognitive complexity. Add "why" or "how" questions to move beyond recall.',
            'priority': 'Medium',
            'icon': ''
        })
    elif difficulty == 'Hard':
        improvements.append({
            'category': 'Difficulty',
            'suggestion': 'Simplify language or break into sub-questions to reduce cognitive load.',
            'priority': 'High',
            'icon': ''
        })
    
    # Cognitive level improvements
    if cognitive_level in ['Remember', 'Understand']:
        improvements.append({
            'category': 'Cognitive Level',
            'suggestion': f'Current level: {cognitive_level}. Consider moving to Apply/Analyze for deeper learning.',
            'priority': 'Medium',
            'icon': ''
        })
    
    # Text metrics improvements
    if metrics['word_count'] > 40:
        improvements.append({
            'category': 'Clarity',
            'suggestion': f'Question is long ({metrics["word_count"]} words). Consider breaking into shorter, clearer questions.',
            'priority': 'Medium',
            'icon': ''
        })
    elif metrics['word_count'] < 10:
        improvements.append({
            'category': 'Clarity',
            'suggestion': 'Question is very short. Add context or details to make it clearer.',
            'priority': 'Low',
            'icon': ''
        })
    
    # Sentence complexity
    if metrics['sentence_count'] > 3:
        improvements.append({
            'category': 'Structure',
            'suggestion': 'Multiple sentences detected. Consider simplifying or using bullet points.',
            'priority': 'Low',
            'icon': ''
        })
    
    return improvements

# Main UI
st.markdown("<h1 class='main-header'> Assessment Intelligence Platform</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <span class='milestone-badge m1-badge'> Milestone 1: ML Analytics</span>
    <span class='milestone-badge m2-badge'> Milestone 2: Agentic AI</span>
</div>
""", unsafe_allow_html=True)

st.markdown("*Unified Platform: From ML-Based Analysis to Autonomous Assessment Improvement*")

# Sidebar
with st.sidebar:
    st.markdown("###  Navigation")
    mode = st.radio(
        "Select Mode:",
        [" Question Analysis", " Batch Processing", " AI Insights", " Knowledge Base", " About"]
    )

# Main content
if mode == " Question Analysis":
    st.markdown("<h2 class='sub-header'>Single Question Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        question_text = st.text_area("Enter question text:", height=100, placeholder="Type your exam question here...")
        subject = st.selectbox("Subject:", ["Mathematics", "Physics", "Computer Science", "Engineering"])
    
    with col2:
        cognitive_level = st.selectbox("Cognitive Level (Bloom's):", 
                                       ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"])
        grade_level = st.slider("Grade Level:", 1, 12, 10)
    
    if st.button(" Analyze Question", key="analyze_btn", use_container_width=True):
        if question_text:
            model, vectorizer = load_models()
            if model and vectorizer:
                # Milestone 1: Predict difficulty
                difficulty, confidence = predict_difficulty(question_text, model, vectorizer)
                
                # Display M1 Results
                st.markdown("<h3> Milestone 1: ML-Based Analysis</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Difficulty", difficulty)
                with col2:
                    st.metric("Confidence", f"{max(confidence.values()):.1%}")
                with col3:
                    st.metric("Cognitive Level", cognitive_level)
                
                # Confidence distribution
                st.bar_chart(confidence)
                
                # Milestone 2: Generate improvements
                st.markdown("<h3> Milestone 2: AI-Powered Improvements</h3>", unsafe_allow_html=True)
                
                improvements = generate_improvements(question_text, difficulty, cognitive_level)
                
                if improvements:
                    for imp in improvements:
                        priority_color = "" if imp['priority'] == 'High' else ("" if imp['priority'] == 'Medium' else "")
                        st.markdown(f"""
                        <div class='improvement-box'>
                        {imp['icon']} **{imp['category']}** ({priority_color} {imp['priority']})<br>
                        {imp['suggestion']}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Metrics
                st.markdown("<h3> Question Metrics</h3>", unsafe_allow_html=True)
                metrics = analyze_question_metrics(question_text)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Word Count", metrics['word_count'])
                with col2:
                    st.metric("Sentences", metrics['sentence_count'])
                with col3:
                    st.metric("Avg Word Length", f"{metrics['avg_word_length']:.1f}")
                with col4:
                    st.metric("Readability", metrics['readability_estimate'])

elif mode == " Batch Processing":
    st.markdown("<h2 class='sub-header'>Batch Question Analysis</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV with questions:", type="csv")
    
    if uploaded_file and st.button("⚡ Process Batch", use_container_width=True):
        df = pd.read_csv(uploaded_file)
        model, vectorizer = load_models()
        
        if model and vectorizer:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, row in df.iterrows():
                question = row.get('question_text', '')
                difficulty, confidence = predict_difficulty(question, model, vectorizer)
                
                results.append({
                    'question': question[:50],
                    'difficulty': difficulty,
                    'confidence': max(confidence.values()) if confidence else 0
                })
                
                progress_bar.progress((idx + 1) / len(df))
                status_text.text(f"Processing: {idx + 1}/{len(df)}")
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            st.markdown("<h3> Summary Statistics</h3>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Questions", len(results_df))
            with col2:
                st.metric("Avg Confidence", f"{results_df['confidence'].mean():.1%}")
            with col3:
                easy_count = len(results_df[results_df['difficulty'] == 'Easy'])
                st.metric("Easy Questions", easy_count)
            with col4:
                hard_count = len(results_df[results_df['difficulty'] == 'Hard'])
                st.metric("Hard Questions", hard_count)
            
            # Difficulty distribution
            st.bar_chart(results_df['difficulty'].value_counts())
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(" Download Results", csv, "assessment_results.csv", "text/csv")

elif mode == " AI Insights":
    st.markdown("<h2 class='sub-header'>AI-Powered Assessment Insights</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ###  Milestone 2: Agentic AI Features
    
    This section demonstrates autonomous AI reasoning for assessment improvement:
    
    **Key Capabilities:**
    -  Autonomous question analysis
    -  Intelligent improvement suggestions
    -  Pedagogical knowledge retrieval
    -  Learning objective alignment
    -  Assessment quality metrics
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("####  Question Analysis Workflow")
        st.markdown("""
        1. **Analyze** - Extract metrics & predict difficulty
        2. **Retrieve** - Query pedagogical knowledge base
        3. **Reason** - Generate improvement suggestions
        4. **Validate** - Check quality & alignment
        5. **Output** - Structured recommendations
        """)
    
    with col2:
        st.markdown("####  Supported Features")
        st.markdown("""
        - Bloom's Taxonomy alignment
        - Difficulty calibration
        - Learning gap identification
        - Question discrimination analysis
        - Readability assessment
        - Cognitive load evaluation
        """)

elif mode == " Knowledge Base":
    st.markdown("<h2 class='sub-header'>Pedagogical Knowledge Base</h2>", unsafe_allow_html=True)
    
    st.markdown("###  Assessment Design Best Practices")
    
    for topic, content in PEDAGOGICAL_KNOWLEDGE.items():
        with st.expander(f"📌 {topic}"):
            st.info(content)

elif mode == " About":
    st.markdown("<h2 class='sub-header'>About This Platform</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ###  Milestone 1: ML-Based Analytics
        
        **Status**:  Complete
        
        - ML-based difficulty prediction
        - TF-IDF vectorization (5,000 features)
        - Logistic Regression model (31.4% accuracy)
        - Real-time predictions (< 0.1s)
        - Batch processing support
        - CSV export functionality
        """)
    
    with col2:
        st.markdown("""
        ###  Milestone 2: Agentic AI
        
        **Status**:  Complete
        
        - LangGraph-based workflow
        - RAG integration
        - Pedagogical knowledge base
        - Autonomous improvement generation
        - State management
        - Structured recommendations
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ###  Project Statistics
    
    | Metric | Value |
    |--------|-------|
    | Questions Analyzed | 5,000 |
    | Subjects Covered | 4 |
    | ML Models Trained | 3 |
    | Best Accuracy | 31.4% |
    | Knowledge Base Docs | 8 |
    | Deployment Status |  Live |
    
    ###  Links
    
    - **Live Demo**: https://intelligent-exam-question-analysis-agentic-assessment-design-z.streamlit.app/
    - **GitHub**: https://github.com/ArPriCode/INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN
    - **Documentation**: See README.md
    """)
    
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This system provides AI-assisted assessment analysis. 
    Educators should review all suggestions and apply professional judgment.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p> Intelligent Exam Question Analysis & Agentic Assessment Design</p>
    <p>Made with  for Education | Version 2.0.0 | Status:  Production Ready</p>
</div>
""", unsafe_allow_html=True)
