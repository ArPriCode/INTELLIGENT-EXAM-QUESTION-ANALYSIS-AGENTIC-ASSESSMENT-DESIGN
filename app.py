import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Intelligent Exam Question Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        model = joblib.load('models/difficulty_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

def predict_difficulty(question_text, model, vectorizer):
    """Predict question difficulty"""
    if not question_text.strip():
        return None
    
    # Vectorize the question text
    features = vectorizer.transform([question_text])
    
    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    difficulty_map = {0: "Easy", 1: "Medium", 2: "Hard"}
    
    return {
        'difficulty': difficulty_map[prediction],
        'confidence': max(probabilities),
        'probabilities': {
            'Easy': probabilities[0],
            'Medium': probabilities[1],
            'Hard': probabilities[2]
        }
    }

def analyze_text_metrics(text):
    """Analyze basic text metrics"""
    words = text.split()
    sentences = text.split('.')
    
    return {
        'word_count': len(words),
        'character_count': len(text),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">Intelligent Exam Question Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">ML-Based Question Difficulty Prediction</p>', unsafe_allow_html=True)
    
    # Load models
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.error("Models not found! Please run `python train_model.py` first.")
        st.info("Make sure you have the trained models in the `models/` directory.")
        return
    
    # Sidebar
    st.sidebar.header("Navigation")
    tab_selection = st.sidebar.radio(
        "Choose Analysis Type:",
        ["Single Question Analysis", "Batch Analysis", "Model Information"]
    )
    
    if tab_selection == "Single Question Analysis":
        single_question_analysis(model, vectorizer)
    elif tab_selection == "Batch Analysis":
        batch_analysis(model, vectorizer)
    else:
        model_information()

def single_question_analysis(model, vectorizer):
    """Single question analysis interface"""
    st.header("Single Question Analysis")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question_text = st.text_area(
            "Enter your exam question:",
            height=150,
            placeholder="Example: Solve the quadratic equation x² + 5x + 6 = 0"
        )
        
        # Optional metadata
        with st.expander("Optional Metadata"):
            col_meta1, col_meta2 = st.columns(2)
            with col_meta1:
                subject = st.selectbox(
                    "Subject:",
                    ["", "Mathematics", "Physics", "Computer Science", "Engineering"]
                )
                grade_level = st.selectbox(
                    "Grade Level:",
                    ["", "High School", "Undergraduate", "Graduate"]
                )
            with col_meta2:
                question_type = st.selectbox(
                    "Question Type:",
                    ["", "Multiple Choice", "Short Answer", "Essay", "Problem Solving"]
                )
    
    with col2:
        st.markdown("### Quick Stats")
        if question_text:
            metrics = analyze_text_metrics(question_text)
            st.metric("Word Count", metrics['word_count'])
            st.metric("Characters", metrics['character_count'])
            st.metric("Sentences", metrics['sentence_count'])
            st.metric("Avg Word Length", f"{metrics['avg_word_length']:.1f}")
    
    # Analysis button
    if st.button("Analyze Question", type="primary", use_container_width=True):
        if not question_text.strip():
            st.warning("Please enter a question to analyze.")
            return
        
        with st.spinner("Analyzing question..."):
            result = predict_difficulty(question_text, model, vectorizer)
            
            if result:
                # Results section
                st.markdown("---")
                st.header("Analysis Results")
                
                # Main prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Predicted Difficulty</h3>
                        <h2>{result['difficulty']}</h2>
                        <p>Confidence: {result['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Confidence Score</h3>
                        <h2>{result['confidence']:.1%}</h2>
                        <p>Model certainty</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    metrics = analyze_text_metrics(question_text)
                    complexity_score = min(100, (metrics['word_count'] * 2 + metrics['avg_word_length'] * 5))
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Complexity Score</h3>
                        <h2>{complexity_score:.0f}/100</h2>
                        <p>Text complexity</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability distribution chart
                st.subheader("Probability Distribution")
                prob_df = pd.DataFrame({
                    'Difficulty': list(result['probabilities'].keys()),
                    'Probability': list(result['probabilities'].values())
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Difficulty', 
                    y='Probability',
                    color='Difficulty',
                    color_discrete_map={'Easy': '#2ecc71', 'Medium': '#f39c12', 'Hard': '#e74c3c'},
                    title="Difficulty Prediction Probabilities"
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("Recommendations")
                if result['difficulty'] == "Easy":
                    st.markdown("""
                    <div class="success-box">
                        <strong>Easy Question Detected</strong><br>
                        • Suitable for introductory assessments<br>
                        • Good for building student confidence<br>
                        • Consider for warm-up questions
                    </div>
                    """, unsafe_allow_html=True)
                elif result['difficulty'] == "Medium":
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Medium Difficulty Question</strong><br>
                        • Appropriate for regular assessments<br>
                        • Good balance of challenge and accessibility<br>
                        • Suitable for most students
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 0.25rem; padding: 1rem; margin: 1rem 0;">
                        <strong>Hard Question Detected</strong><br>
                        • Challenging question for advanced students<br>
                        • Consider for final exams or advanced courses<br>
                        • May require additional preparation time
                    </div>
                    """, unsafe_allow_html=True)

def batch_analysis(model, vectorizer):
    """Batch analysis interface"""
    st.header("Batch Question Analysis")
    
    st.info("Upload a CSV file with a 'question_text' column to analyze multiple questions at once.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV file should have a 'question_text' column"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'question_text' not in df.columns:
                st.error("CSV file must contain a 'question_text' column.")
                return
            
            st.success(f"Loaded {len(df)} questions from CSV file.")
            
            # Show preview
            with st.expander("Preview Data"):
                st.dataframe(df.head())
            
            if st.button("Analyze All Questions", type="primary"):
                with st.spinner("Analyzing questions..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, question in enumerate(df['question_text']):
                        if pd.notna(question) and question.strip():
                            result = predict_difficulty(str(question), model, vectorizer)
                            if result:
                                results.append({
                                    'question_text': question,
                                    'predicted_difficulty': result['difficulty'],
                                    'confidence': result['confidence'],
                                    'easy_prob': result['probabilities']['Easy'],
                                    'medium_prob': result['probabilities']['Medium'],
                                    'hard_prob': result['probabilities']['Hard']
                                })
                            else:
                                results.append({
                                    'question_text': question,
                                    'predicted_difficulty': 'Error',
                                    'confidence': 0,
                                    'easy_prob': 0,
                                    'medium_prob': 0,
                                    'hard_prob': 0
                                })
                        progress_bar.progress((i + 1) / len(df))
                    
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.header("Batch Analysis Results")
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Questions", len(results_df))
                    with col2:
                        easy_count = len(results_df[results_df['predicted_difficulty'] == 'Easy'])
                        st.metric("Easy Questions", easy_count)
                    with col3:
                        medium_count = len(results_df[results_df['predicted_difficulty'] == 'Medium'])
                        st.metric("Medium Questions", medium_count)
                    with col4:
                        hard_count = len(results_df[results_df['predicted_difficulty'] == 'Hard'])
                        st.metric("Hard Questions", hard_count)
                    
                    # Distribution chart
                    difficulty_counts = results_df['predicted_difficulty'].value_counts()
                    fig = px.pie(
                        values=difficulty_counts.values,
                        names=difficulty_counts.index,
                        title="Difficulty Distribution",
                        color_discrete_map={'Easy': '#2ecc71', 'Medium': '#f39c12', 'Hard': '#e74c3c'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="question_analysis_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def model_information():
    """Display model information"""
    st.header("Model Information")
    
    # Model details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.info("""
        **Algorithm**: Logistic Regression  
        **Features**: TF-IDF Vectorization (1,500 features)  
        **Training Data**: 5,000 exam questions  
        **Accuracy**: ~72%  
        **Classes**: Easy, Medium, Hard
        """)
        
        st.subheader("Performance Metrics")
        st.info("""
        **Precision**: 0.72 (overall)  
        **Recall**: 0.71 (overall)  
        **F1-Score**: 0.71 (overall)  
        **Cross-Validation**: Stratified 80-20 split
        """)
    
    with col2:
        st.subheader("Technical Specifications")
        st.info("""
        **Preprocessing**: Text cleaning, tokenization  
        **Vectorization**: TF-IDF (1-2 grams)  
        **Stop Words**: English  
        **Max Features**: 1,500  
        **Model Type**: Multiclass classification
        """)
        
        st.subheader("Dataset Information")
        st.info("""
        **Total Questions**: 5,000  
        **Subjects**: Mathematics, Physics, CS, Engineering  
        **Difficulty Distribution**: Balanced  
        **Source**: Educational assessment data
        """)
    
    # Model insights
    st.subheader("Model Insights")
    st.warning("""
    **Note**: This model provides difficulty predictions based on text analysis. 
    Results should be used as guidance alongside educator expertise. 
    The model may have limitations with domain-specific terminology or context.
    """)
    
    # Usage tips
    st.subheader("Usage Tips")
    st.markdown("""
    1. **Question Quality**: Ensure questions are well-formatted and clear
    2. **Context Matters**: Consider subject domain and student level
    3. **Confidence Scores**: Pay attention to prediction confidence
    4. **Batch Processing**: Use CSV upload for analyzing multiple questions
    5. **Validation**: Always validate predictions with educational expertise
    """)

if __name__ == "__main__":
    main()