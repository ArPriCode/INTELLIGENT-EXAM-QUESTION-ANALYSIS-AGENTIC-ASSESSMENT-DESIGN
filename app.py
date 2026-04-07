import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Page config
st.set_page_config(
    page_title="Exam Question Analysis",
    page_icon="📚",
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
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/difficulty_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except:
        return None, None

def predict_difficulty(question_text, model, vectorizer):
    """Predict difficulty level of question"""
    if model is None or vectorizer is None:
        return None, None
    
    # Transform text
    features = vectorizer.transform([question_text])
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return prediction, probability

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Intelligent Exam Question Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">AI-Driven Educational Analytics System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=100)
        st.title("Navigation")
        page = st.radio("Select Module", 
                       ["Question Analysis", "Batch Analysis", "Model Info"])
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This system analyzes exam questions using ML & NLP techniques to predict difficulty levels.")
    
    # Load models
    model, vectorizer = load_models()
    
    if page == "Question Analysis":
        st.markdown('<h2 class="sub-header">Single Question Analysis</h2>', unsafe_allow_html=True)
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            question_text = st.text_area(
                "Enter Exam Question:",
                height=150,
                placeholder="Type or paste your exam question here..."
            )
            
            # Additional metadata
            st.markdown("#### Optional Metadata")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                subject = st.selectbox("Subject", ["Mathematics", "Science", "English", "History", "Other"])
            with col_b:
                grade = st.selectbox("Grade Level", ["6", "7", "8", "9", "10", "11", "12"])
            with col_c:
                q_type = st.selectbox("Question Type", ["MCQ", "Short Answer", "Essay", "Problem Solving"])
        
        with col2:
            st.markdown("#### Quick Stats")
            if question_text:
                word_count = len(question_text.split())
                char_count = len(question_text)
                st.metric("Word Count", word_count)
                st.metric("Character Count", char_count)
        
        # Analyze button
        if st.button("🔍 Analyze Question", type="primary", use_container_width=True):
            if not question_text:
                st.error("Please enter a question to analyze!")
            elif model is None:
                st.warning("⚠️ Model not found! Please train the model first.")
                st.info("Run the training script to generate models.")
            else:
                with st.spinner("Analyzing question..."):
                    prediction, probability = predict_difficulty(question_text, model, vectorizer)
                    
                    # Results
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("### 📊 Analysis Results")
                    
                    # Difficulty prediction
                    col1, col2, col3 = st.columns(3)
                    
                    difficulty_map = {0: "Easy", 1: "Medium", 2: "Hard"}
                    color_map = {0: "🟢", 1: "🟡", 2: "🔴"}
                    
                    with col1:
                        st.markdown(f'<div class="metric-card"><h3>{color_map[prediction]} {difficulty_map[prediction]}</h3><p>Predicted Difficulty</p></div>', unsafe_allow_html=True)
                    
                    with col2:
                        confidence = max(probability) * 100
                        st.markdown(f'<div class="metric-card"><h3>{confidence:.1f}%</h3><p>Confidence Score</p></div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f'<div class="metric-card"><h3>{word_count}</h3><p>Words</p></div>', unsafe_allow_html=True)
                    
                    # Probability distribution
                    st.markdown("#### Difficulty Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Difficulty': ['Easy', 'Medium', 'Hard'],
                        'Probability': probability
                    })
                    st.bar_chart(prob_df.set_index('Difficulty'))
                    
                    # Recommendations
                    st.markdown("#### 💡 Recommendations")
                    if prediction == 0:
                        st.success("✅ This question is suitable for introductory assessments or warm-up exercises.")
                    elif prediction == 1:
                        st.info("ℹ️ This question has moderate difficulty. Good for regular assessments.")
                    else:
                        st.warning("⚠️ This is a challenging question. Consider for advanced students or final exams.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Batch Analysis":
        st.markdown('<h2 class="sub-header">Batch Question Analysis</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload CSV file with questions", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("### Preview of uploaded data:")
            st.dataframe(df.head())
            
            if 'question' in df.columns:
                if st.button("Analyze All Questions"):
                    if model is None:
                        st.error("Model not found!")
                    else:
                        with st.spinner("Analyzing questions..."):
                            predictions = []
                            for question in df['question']:
                                pred, prob = predict_difficulty(question, model, vectorizer)
                                predictions.append(pred)
                            
                            df['predicted_difficulty'] = predictions
                            df['difficulty_label'] = df['predicted_difficulty'].map({0: "Easy", 1: "Medium", 2: "Hard"})
                            
                            st.success("Analysis complete!")
                            st.dataframe(df)
                            
                            # Summary stats
                            st.markdown("### 📈 Summary Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Easy Questions", len(df[df['predicted_difficulty']==0]))
                            with col2:
                                st.metric("Medium Questions", len(df[df['predicted_difficulty']==1]))
                            with col3:
                                st.metric("Hard Questions", len(df[df['predicted_difficulty']==2]))
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button("Download Results", csv, "analysis_results.csv", "text/csv")
            else:
                st.error("CSV must contain a 'question' column!")
    
    else:  # Model Info
        st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
        
        if model is None:
            st.warning("⚠️ No trained model found!")
            st.info("Please run the training script first to generate the model.")
            
            with st.expander("📝 Training Instructions"):
                st.code("""
# Run the training script
python train_model.py

# This will generate:
# - models/difficulty_model.pkl
# - models/tfidf_vectorizer.pkl
                """)
        else:
            st.success("✅ Model loaded successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Model Details")
                st.write(f"**Model Type:** {type(model).__name__}")
                st.write(f"**Features:** TF-IDF Vectorization")
                st.write(f"**Classes:** Easy, Medium, Hard")
            
            with col2:
                st.markdown("### Performance Metrics")
                st.info("Train the model to see performance metrics")

if __name__ == "__main__":
    main()
