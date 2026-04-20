import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# Create models directory
os.makedirs('models', exist_ok=True)

def load_data():
    """Load the actual dataset"""
    try:
        # Try data folder first, then root directory
        if os.path.exists('data/question_ans_analysis.csv'):
            df = pd.read_csv('data/question_ans_analysis.csv')
        else:
            df = pd.read_csv('question_ans_analysis.csv')
        print(f" Loaded dataset: {len(df)} questions")
        return df
    except FileNotFoundError:
        print(" question_ans_analysis.csv not found in data/ or root directory!")
        return None

def train_model():
    """Train the difficulty prediction model"""
    print("Loading dataset...")
    df = load_data()
    
    if df is None:
        return None, None
    
    # Map difficulty labels to numeric
    difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
    df['difficulty_numeric'] = df['difficulty_label'].map(difficulty_map)
    
    print(f"\nDataset size: {len(df)} questions")
    print(f"Difficulty distribution:\n{df['difficulty_label'].value_counts()}")
    print(f"\nSubject distribution:\n{df['subject'].value_counts()}")
    
    # Create additional features
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Prepare features
    text_features = df['question_text']
    numeric_features = df[['readability_score', 'word_count', 'sentence_count', 
                          'correct_percentage', 'learning_gap_score', 'discrimination_index']]
    
    # Split data
    X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
        text_features, numeric_features, df['difficulty_numeric'], 
        test_size=0.2, random_state=42, stratify=df['difficulty_numeric']
    )
    
    print("\nTraining TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    
    # Combine features
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_tfidf, X_train_num_scaled])
    X_test_combined = hstack([X_test_tfidf, X_test_num_scaled])
    
    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"\n{name}:")
        model.fit(X_train_combined, y_train)
        y_pred = model.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred, target_names=['Easy', 'Medium', 'Hard']))
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with accuracy {best_score:.3f}")
    
    # Save best model, vectorizer, and scaler
    print("\nSaving models...")
    joblib.dump(best_model, 'models/difficulty_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(" Models saved successfully!")
    print("   - models/difficulty_model.pkl")
    print("   - models/tfidf_vectorizer.pkl")
    print("   - models/scaler.pkl")
    
    return best_model, vectorizer

if __name__ == "__main__":
    train_model()
