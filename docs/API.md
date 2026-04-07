#  API Documentation

Complete API reference for the Exam Question Analysis system.

---

##  Overview

This document provides detailed information about the internal APIs and functions used in the system.

---

## 🧠 Model API

### Load Models

```python
import joblib

def load_models():
    """
    Load trained model and vectorizer from disk.
    
    Returns:
        tuple: (model, vectorizer) or (None, None) if files not found
    """
    try:
        model = joblib.load('models/difficulty_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None
```

**Usage:**
```python
model, vectorizer = load_models()
if model is None:
    print("Models not found. Please train first.")
```

---

### Predict Difficulty

```python
def predict_difficulty(question_text, model, vectorizer):
    """
    Predict difficulty level of a question.
    
    Args:
        question_text (str): The exam question text
        model: Trained classifier model
        vectorizer: Fitted TF-IDF vectorizer
    
    Returns:
        tuple: (prediction, probability_array)
            - prediction (int): 0=Easy, 1=Medium, 2=Hard
            - probability_array (np.array): [P(easy), P(medium), P(hard)]
    
    Example:
        >>> pred, prob = predict_difficulty("What is 2+2?", model, vectorizer)
        >>> print(f"Difficulty: {pred}, Confidence: {max(prob):.2f}")
        Difficulty: 0, Confidence: 0.85
    """
    if model is None or vectorizer is None:
        return None, None
    
    # Transform text to features
    features = vectorizer.transform([question_text])
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return prediction, probability
```

**Response Format:**
```python
{
    'prediction': 0,  # 0=Easy, 1=Medium, 2=Hard
    'probability': [0.85, 0.10, 0.05]  # [Easy, Medium, Hard]
}
```

---

##  Data Processing API

### Load Dataset

```python
def load_data(filepath='question_ans_analysis.csv'):
    """
    Load exam questions dataset from CSV.
    
    Args:
        filepath (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset or None if error
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    try:
        df = pd.read_csv(filepath)
        print(f" Loaded {len(df)} questions")
        return df
    except FileNotFoundError:
        print(f" File not found: {filepath}")
        return None
```

---

### Preprocess Text

```python
def preprocess_text(text):
    """
    Clean and preprocess question text.
    
    Args:
        text (str): Raw question text
    
    Returns:
        str: Cleaned text
    
    Example:
        >>> preprocess_text("What is 2+2?  ")
        'what is 2+2'
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
```

---

##  Training API

### Train Model

```python
def train_model(data_path='question_ans_analysis.csv'):
    """
    Train difficulty prediction model.
    
    Args:
        data_path (str): Path to training data CSV
    
    Returns:
        tuple: (best_model, vectorizer, metrics)
    
    Example:
        >>> model, vec, metrics = train_model()
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    # Load data
    df = load_data(data_path)
    
    # Prepare features and labels
    X = df['question_text']
    y = df['difficulty_label'].map({'easy': 0, 'medium': 1, 'hard': 2})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train vectorizer
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(n_estimators=100)
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        score = model.score(X_test_vec, y_test)
        if score > best_score:
            best_score = score
            best_model = model
    
    # Calculate metrics
    y_pred = best_model.predict(X_test_vec)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    return best_model, vectorizer, metrics
```

---

## 📈 Evaluation API

### Calculate Metrics

```python
def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
    
    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, 
        recall_score, f1_score, confusion_matrix
    )
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
```

---

##  Streamlit UI API

### Display Results

```python
def display_prediction_results(prediction, probability, question_text):
    """
    Display prediction results in Streamlit UI.
    
    Args:
        prediction (int): Predicted difficulty (0, 1, or 2)
        probability (array): Probability distribution
        question_text (str): Original question
    """
    import streamlit as st
    
    difficulty_map = {0: "Easy", 1: "Medium", 2: "Hard"}
    color_map = {0: "🟢", 1: "🟡", 2: "🔴"}
    
    # Display difficulty
    st.markdown(f"## {color_map[prediction]} {difficulty_map[prediction]}")
    
    # Display confidence
    confidence = max(probability) * 100
    st.metric("Confidence", f"{confidence:.1f}%")
    
    # Display probability chart
    import pandas as pd
    prob_df = pd.DataFrame({
        'Difficulty': ['Easy', 'Medium', 'Hard'],
        'Probability': probability
    })
    st.bar_chart(prob_df.set_index('Difficulty'))
```

---

##  Batch Processing API

### Batch Predict

```python
def batch_predict(questions, model, vectorizer):
    """
    Predict difficulty for multiple questions.
    
    Args:
        questions (list): List of question texts
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
    
    Returns:
        pd.DataFrame: Results with predictions
    
    Example:
        >>> questions = ["What is 2+2?", "Explain relativity"]
        >>> results = batch_predict(questions, model, vectorizer)
        >>> print(results)
    """
    import pandas as pd
    
    results = []
    for question in questions:
        pred, prob = predict_difficulty(question, model, vectorizer)
        results.append({
            'question': question,
            'prediction': pred,
            'difficulty': ['Easy', 'Medium', 'Hard'][pred],
            'confidence': max(prob),
            'prob_easy': prob[0],
            'prob_medium': prob[1],
            'prob_hard': prob[2]
        })
    
    return pd.DataFrame(results)
```

---

##  Model Persistence API

### Save Model

```python
def save_model(model, vectorizer, model_path='models/'):
    """
    Save trained model and vectorizer to disk.
    
    Args:
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        model_path (str): Directory to save models
    
    Returns:
        bool: True if successful
    """
    import os
    import joblib
    
    os.makedirs(model_path, exist_ok=True)
    
    try:
        joblib.dump(model, f'{model_path}/difficulty_model.pkl')
        joblib.dump(vectorizer, f'{model_path}/tfidf_vectorizer.pkl')
        print(" Models saved successfully")
        return True
    except Exception as e:
        print(f" Error saving models: {e}")
        return False
```

---

##  Utility Functions

### Get Question Statistics

```python
def get_question_stats(question_text):
    """
    Calculate statistics for a question.
    
    Args:
        question_text (str): Question text
    
    Returns:
        dict: Statistics dictionary
    """
    import re
    
    words = question_text.split()
    sentences = re.split(r'[.!?]+', question_text)
    
    return {
        'word_count': len(words),
        'char_count': len(question_text),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0
    }
```

---

##  Response Formats

### Standard Response

```json
{
    "status": "success",
    "data": {
        "prediction": 0,
        "difficulty": "Easy",
        "confidence": 0.85,
        "probabilities": {
            "easy": 0.85,
            "medium": 0.10,
            "hard": 0.05
        },
        "metadata": {
            "word_count": 5,
            "char_count": 15,
            "processing_time_ms": 45
        }
    }
}
```

### Error Response

```json
{
    "status": "error",
    "error": {
        "code": "MODEL_NOT_FOUND",
        "message": "Trained model not found. Please train the model first.",
        "details": "models/difficulty_model.pkl does not exist"
    }
}
```

---

## 🔐 Authentication (Future)

### API Key Authentication

```python
def authenticate_request(api_key):
    """
    Authenticate API request using API key.
    
    Args:
        api_key (str): API key from request header
    
    Returns:
        bool: True if authenticated
    """
    import os
    
    valid_key = os.getenv('API_KEY')
    return api_key == valid_key
```

---

##  Usage Examples

### Complete Workflow

```python
# 1. Load models
model, vectorizer = load_models()

# 2. Single prediction
question = "What is the capital of France?"
pred, prob = predict_difficulty(question, model, vectorizer)
print(f"Difficulty: {['Easy', 'Medium', 'Hard'][pred]}")
print(f"Confidence: {max(prob):.2%}")

# 3. Batch prediction
questions = [
    "What is 2+2?",
    "Explain quantum mechanics",
    "Calculate derivative of x^2"
]
results = batch_predict(questions, model, vectorizer)
print(results)

# 4. Get statistics
stats = get_question_stats(question)
print(f"Word count: {stats['word_count']}")
```

---

##  Error Handling

### Common Errors

```python
class ModelNotFoundError(Exception):
    """Raised when model files are not found"""
    pass

class InvalidInputError(Exception):
    """Raised when input validation fails"""
    pass

def safe_predict(question_text, model, vectorizer):
    """
    Safely predict with error handling.
    """
    try:
        if not question_text or len(question_text) < 5:
            raise InvalidInputError("Question too short")
        
        if model is None:
            raise ModelNotFoundError("Model not loaded")
        
        return predict_difficulty(question_text, model, vectorizer)
    
    except InvalidInputError as e:
        print(f"Input error: {e}")
        return None, None
    except ModelNotFoundError as e:
        print(f"Model error: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None
```

---

##  Support

For API questions or issues:
- Open GitHub issue
- Check documentation
- Contact maintainers

---

**API Version: 1.0.0**
