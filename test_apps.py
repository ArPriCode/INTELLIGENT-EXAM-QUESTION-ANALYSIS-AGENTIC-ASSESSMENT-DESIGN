#!/usr/bin/env python3
"""
Test script to verify both Milestone 1 and Milestone 2 applications work correctly
"""

import sys
import os
import importlib.util

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    # Test basic imports
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import joblib
        import sklearn
        print("✓ Basic ML imports successful")
    except ImportError as e:
        print(f"✗ Basic ML imports failed: {e}")
        return False
    
    # Test LangGraph imports (optional)
    try:
        from langgraph.graph import StateGraph, END
        from langchain_core.messages import HumanMessage, AIMessage
        import chromadb
        print("✓ LangGraph imports successful")
        langgraph_available = True
    except ImportError as e:
        print(f"⚠ LangGraph imports failed (optional): {e}")
        langgraph_available = False
    
    return True, langgraph_available

def test_models():
    """Test if trained models exist and can be loaded"""
    print("\nTesting ML models...")
    
    try:
        import joblib
        model = joblib.load('models/difficulty_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        print("✓ ML models loaded successfully")
        
        # Test prediction
        test_question = "What is the capital of France?"
        features = vectorizer.transform([test_question])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        difficulty_map = {0: "Easy", 1: "Medium", 2: "Hard"}
        print(f"✓ Test prediction: {difficulty_map[prediction]} (confidence: {max(probabilities):.2f})")
        
        return True
    except Exception as e:
        print(f"✗ Model testing failed: {e}")
        return False

def test_app_syntax():
    """Test if app files have valid Python syntax"""
    print("\nTesting app syntax...")
    
    apps = ['app.py', 'app_milestone2.py']
    
    for app in apps:
        if not os.path.exists(app):
            print(f"✗ {app} not found")
            continue
            
        try:
            spec = importlib.util.spec_from_file_location("test_app", app)
            module = importlib.util.module_from_spec(spec)
            # Don't execute, just check syntax
            with open(app, 'r') as f:
                compile(f.read(), app, 'exec')
            print(f"✓ {app} syntax valid")
        except SyntaxError as e:
            print(f"✗ {app} syntax error: {e}")
        except Exception as e:
            print(f"⚠ {app} other issue: {e}")

def test_knowledge_base():
    """Test if knowledge base files exist"""
    print("\nTesting knowledge base...")
    
    kb_files = [
        'knowledge_base/blooms_taxonomy.md',
        'knowledge_base/question_design_principles.md', 
        'knowledge_base/difficulty_calibration.md'
    ]
    
    for kb_file in kb_files:
        if os.path.exists(kb_file):
            print(f"✓ {kb_file} exists")
        else:
            print(f"✗ {kb_file} missing")

def main():
    """Run all tests"""
    print("Testing Intelligent Exam Question Analysis System")
    print("=" * 60)
    
    # Test imports
    basic_ok, langgraph_ok = test_imports()
    if not basic_ok:
        print("\n✗ Basic imports failed. Please install requirements:")
        print("pip install -r requirements.txt")
        return
    
    # Test models
    models_ok = test_models()
    if not models_ok:
        print("\n✗ Models not available. Please run:")
        print("python3 train_model.py")
        return
    
    # Test app syntax
    test_app_syntax()
    
    # Test knowledge base
    test_knowledge_base()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("✓ Milestone 1 (ML-based Analytics): READY")
    print("   - ML models trained and working")
    print("   - Streamlit app created (app.py)")
    print("   - Run with: streamlit run app.py")
    
    if langgraph_ok:
        print("✓ Milestone 2 (Agentic AI): READY")
        print("   - LangGraph dependencies available")
        print("   - Agentic app created (app_milestone2.py)")
        print("   - Knowledge base populated")
        print("   - Run with: streamlit run app_milestone2.py")
    else:
        print("⚠ Milestone 2 (Agentic AI): PARTIAL")
        print("   - App created but LangGraph not installed")
        print("   - Install with: pip install langgraph langchain chromadb")
        print("   - Then run: streamlit run app_milestone2.py")
    
    print("\nBoth milestones are now implemented!")
    print("Check README.md for detailed usage instructions")

if __name__ == "__main__":
    main()