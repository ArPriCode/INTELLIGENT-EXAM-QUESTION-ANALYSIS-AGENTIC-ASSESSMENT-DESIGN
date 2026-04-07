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
        df = pd.read_csv('question_ans_analysis.csv')
        print(f"✅ Loaded dataset: {len(df)} questions")
        return df
    except FileNotFoundError:
        print("❌ question_ans_analysis.csv not found!")
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['question_text'], df['difficulty_numeric'], 
        test_size=0.2, random_state=42, stratify=df['difficulty_numeric']
    )
    
    print("\nTraining TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"\n{name}:")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred, target_names=['Easy', 'Medium', 'Hard']))
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with accuracy {best_score:.3f}")
    
    # Save best model and vectorizer
    print("\nSaving models...")
    joblib.dump(best_model, 'models/difficulty_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    print("✅ Models saved successfully!")
    print("   - models/difficulty_model.pkl")
    print("   - models/tfidf_vectorizer.pkl")
    
    return best_model, vectorizer

if __name__ == "__main__":
    train_model()
