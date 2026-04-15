#!/usr/bin/env python3
"""
Repository Setup Verification Script
Checks if all required files and dependencies are properly configured.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_exists(dirpath, description):
    """Check if a directory exists and print status"""
    exists = os.path.isdir(dirpath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dirpath}")
    return exists

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n📦 Checking Python Dependencies...")
    required_packages = [
        'streamlit',
        'sklearn',
        'pandas',
        'numpy',
        'joblib'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} NOT installed")
            all_installed = False
    
    return all_installed

def main():
    """Main verification function"""
    print("=" * 60)
    print("🔍 Repository Health Check")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 0
    
    # Check essential files
    print("\n📄 Checking Essential Files...")
    essential_files = [
        ('README.md', 'README documentation'),
        ('requirements.txt', 'Python dependencies'),
        ('environment.yml', 'Conda environment'),
        ('.gitignore', 'Git ignore file'),
        ('app.py', 'Main application'),
        ('train_model.py', 'Model training script'),
        ('SETUP.md', 'Setup guide'),
    ]
    
    for filepath, description in essential_files:
        total_checks += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    # Check folder structure
    print("\n📁 Checking Folder Structure...")
    required_dirs = [
        ('data', 'Data directory'),
        ('models', 'Models directory'),
        ('notebooks', 'Notebooks directory'),
        ('scripts', 'Scripts directory'),
        ('docs', 'Documentation directory'),
        ('.streamlit', 'Streamlit config directory'),
    ]
    
    for dirpath, description in required_dirs:
        total_checks += 1
        if check_directory_exists(dirpath, description):
            checks_passed += 1
    
    # Check dependencies
    total_checks += 1
    if check_dependencies():
        checks_passed += 1
    
    # Check data files
    print("\n📊 Checking Data Files...")
    data_files = [
        'data/question_ans_analysis.csv',
        'question_ans_analysis.csv'  # Fallback location
    ]
    
    data_found = False
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"✅ Dataset found: {data_file}")
            data_found = True
            break
    
    if not data_found:
        print("⚠️  Dataset not found. Please add question_ans_analysis.csv to data/ folder")
    
    # Check model files
    print("\n🧠 Checking Model Files...")
    model_files = [
        'models/difficulty_model.pkl',
        'models/tfidf_vectorizer.pkl'
    ]
    
    models_exist = all(os.path.exists(f) for f in model_files)
    if models_exist:
        print("✅ Trained models found")
    else:
        print("⚠️  Models not found. Run 'python train_model.py' to train models")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Summary: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("✅ Repository is properly configured!")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
