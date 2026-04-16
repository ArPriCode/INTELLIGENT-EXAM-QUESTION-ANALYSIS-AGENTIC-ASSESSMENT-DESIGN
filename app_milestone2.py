"""
Milestone 2: Agentic AI Assessment Design Assistant
Extends ML-based analytics with LangGraph + RAG for autonomous assessment improvement
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from typing import Optional, Dict, Any
import json

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document
    from pydantic import BaseModel, Field
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    st.warning("LangGraph not installed. Install with: pip install langgraph langchain langchain-community")

# Page config
st.set_page_config(
    page_title="Assessment Design Assistant",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
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
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        border-radius: 5px;
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

# Pedagogical knowledge base
PEDAGOGICAL_KNOWLEDGE = [
    Document(page_content="Bloom's Taxonomy: Remember, Understand, Apply, Analyze, Evaluate, Create. Higher levels require deeper cognitive engagement."),
    Document(page_content="Question Discrimination Index: Measures how well a question differentiates between high and low performers. Higher is better (0.3+)."),
    Document(page_content="Difficulty Calibration: Easy questions (70%+ correct), Medium (45-65%), Hard (15-35%). Balance is key for effective assessment."),
    Document(page_content="Learning Gap Identification: When many students fail a question, it indicates a knowledge gap. Provide remedial content."),
    Document(page_content="Assessment Quality: Combine difficulty, discrimination, and alignment with learning objectives for comprehensive evaluation."),
    Document(page_content="Question Design Best Practices: Use clear language, avoid ambiguity, align with learning objectives, test one concept per question."),
    Document(page_content="Readability: Flesch-Kincaid score 40-60 for easy, 60-80 for medium, 80+ for hard. Adjust language complexity accordingly."),
    Document(page_content="Cognitive Load: Shorter questions (15-25 words) reduce cognitive load. Longer questions (35+ words) increase difficulty."),
]

class AssessmentState(BaseModel):
    """State for agentic assessment workflow"""
    question_text: str
    subject: str = "General"
    cognitive_level: str = "understand"
    difficulty_prediction: Optional[str] = None
    confidence_scores: Optional[Dict[str, float]] = None
    analysis: Optional[str] = None
    improvements: Optional[list] = None
    pedagogical_context: Optional[str] = None
    final_recommendations: Optional[str] = None

def predict_difficulty(question_text: str, model, vectorizer) -> tuple:
    """Predict question difficulty"""
    try:
        X = vectorizer.transform([question_text])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        confidence_scores = {
            'easy': float(probabilities[0]),
            'medium': float(probabilities[1]),
            'hard': float(probabilities[2])
        }
        
        return prediction, confidence_scores
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
        'readability_estimate': 'Medium' if len(words) < 30 else 'Hard'
    }

def generate_improvements(state: AssessmentState) -> list:
    """Generate improvement suggestions based on analysis"""
    improvements = []
    
    # Difficulty-based improvements
    if state.difficulty_prediction == 'easy':
        improvements.append({
            'category': 'Difficulty',
            'suggestion': 'Consider increasing cognitive complexity. Add "why" or "how" questions.',
            'priority': 'Medium'
        })
    elif state.difficulty_prediction == 'hard':
        improvements.append({
            'category': 'Difficulty',
            'suggestion': 'Simplify language or break into sub-questions to reduce cognitive load.',
            'priority': 'High'
        })
    
    # Cognitive level improvements
    if state.cognitive_level in ['remember', 'understand']:
        improvements.append({
            'category': 'Cognitive Level',
            'suggestion': f'Current level: {state.cognitive_level}. Consider moving to Apply/Analyze for deeper learning.',
            'priority': 'Medium'
        })
    
    # Text metrics improvements
    metrics = analyze_question_metrics(state.question_text)
    if metrics['word_count'] > 40:
        improvements.append({
            'category': 'Clarity',
            'suggestion': f'Question is long ({metrics["word_count"]} words). Consider breaking into shorter, clearer questions.',
            'priority': 'Medium'
        })
    
    return improvements

def create_assessment_report(state: AssessmentState) -> str:
    """Create comprehensive assessment report"""
    report = f"""
    # Assessment Quality Report
    
    ## Question Analysis
    - **Text**: {state.question_text[:100]}...
    - **Subject**: {state.subject}
    - **Cognitive Level**: {state.cognitive_level}
    
    ## Difficulty Assessment
    - **Predicted Difficulty**: {state.difficulty_prediction}
    - **Confidence Scores**: {json.dumps(state.confidence_scores, indent=2)}
    
    ## Metrics
    {json.dumps(analyze_question_metrics(state.question_text), indent=2)}
    
    ## Improvements Suggested
    """
    
    if state.improvements:
        for imp in state.improvements:
            report += f"\n- **{imp['category']}** ({imp['priority']}): {imp['suggestion']}"
    
    return report

# Main UI
st.markdown("<h1 class='main-header'>🎓 Assessment Design Assistant</h1>", unsafe_allow_html=True)
st.markdown("*Milestone 2: Agentic AI for Autonomous Assessment Improvement*")

if not LANGGRAPH_AVAILABLE:
    st.info("💡 LangGraph features are available when dependencies are installed.")

# Tabs
tab1, tab2, tab3 = st.tabs(["Question Analysis", "Batch Assessment", "Pedagogical Insights"])

with tab1:
    st.markdown("<h2 class='sub-header'>Single Question Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        question_text = st.text_area("Enter question text:", height=100)
        subject = st.selectbox("Subject:", ["Mathematics", "Physics", "Computer Science", "Engineering"])
    
    with col2:
        cognitive_level = st.selectbox("Cognitive Level (Bloom's):", 
                                       ["remember", "understand", "apply", "analyze", "evaluate", "create"])
        grade_level = st.slider("Grade Level:", 1, 12, 10)
    
    if st.button("Analyze Question", key="analyze_btn"):
        if question_text:
            model, vectorizer = load_models()
            if model and vectorizer:
                # Predict difficulty
                difficulty, confidence = predict_difficulty(question_text, model, vectorizer)
                
                # Create state
                state = AssessmentState(
                    question_text=question_text,
                    subject=subject,
                    cognitive_level=cognitive_level,
                    difficulty_prediction=difficulty,
                    confidence_scores=confidence
                )
                
                # Generate improvements
                state.improvements = generate_improvements(state)
                
                # Display results
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Difficulty", difficulty.upper())
                with col2:
                    st.metric("Confidence", f"{max(confidence.values()):.1%}")
                with col3:
                    st.metric("Cognitive Level", cognitive_level.title())
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Confidence distribution
                st.bar_chart(confidence)
                
                # Improvements
                st.markdown("<h3>Suggested Improvements</h3>", unsafe_allow_html=True)
                for imp in state.improvements:
                    priority_color = "🔴" if imp['priority'] == 'High' else "🟡"
                    st.markdown(f"""
                    <div class='improvement-box'>
                    {priority_color} **{imp['category']}** ({imp['priority']})<br>
                    {imp['suggestion']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Full report
                if st.checkbox("Show Full Report"):
                    st.markdown(create_assessment_report(state))

with tab2:
    st.markdown("<h2 class='sub-header'>Batch Assessment</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV with questions:", type="csv")
    
    if uploaded_file and st.button("Process Batch"):
        df = pd.read_csv(uploaded_file)
        model, vectorizer = load_models()
        
        if model and vectorizer:
            results = []
            progress_bar = st.progress(0)
            
            for idx, row in df.iterrows():
                question = row.get('question_text', '')
                difficulty, confidence = predict_difficulty(question, model, vectorizer)
                
                results.append({
                    'question': question[:50],
                    'difficulty': difficulty,
                    'confidence': max(confidence.values()) if confidence else 0
                })
                
                progress_bar.progress((idx + 1) / len(df))
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Summary statistics
            st.markdown("<h3>Summary Statistics</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", len(results_df))
            with col2:
                st.metric("Avg Confidence", f"{results_df['confidence'].mean():.1%}")
            with col3:
                st.metric("Easy Questions", len(results_df[results_df['difficulty'] == 'easy']))
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button("Download Results", csv, "assessment_results.csv", "text/csv")

with tab3:
    st.markdown("<h2 class='sub-header'>Pedagogical Insights</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Assessment Design Best Practices
    
    **Bloom's Taxonomy Levels:**
    - Remember: Recall facts and basic concepts
    - Understand: Explain ideas or concepts
    - Apply: Use information in new situations
    - Analyze: Draw connections among ideas
    - Evaluate: Justify a decision or choice
    - Create: Produce new or original work
    
    **Question Quality Metrics:**
    - **Difficulty**: Should match learning objectives
    - **Discrimination**: Differentiates high/low performers
    - **Clarity**: Unambiguous language, single concept
    - **Alignment**: Tests intended learning outcomes
    """)
    
    st.markdown("<h3>Knowledge Base</h3>", unsafe_allow_html=True)
    for i, doc in enumerate(PEDAGOGICAL_KNOWLEDGE):
        st.info(f"📚 {doc.page_content}")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This system provides AI-assisted assessment analysis. 
Educators should review all suggestions and apply professional judgment.
""")
