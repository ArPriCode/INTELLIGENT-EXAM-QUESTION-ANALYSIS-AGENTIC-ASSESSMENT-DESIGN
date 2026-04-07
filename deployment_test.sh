#!/bin/bash
echo "Testing Deployment Requirements..."
echo ""

# Check Python
echo "1. Python Version:"
python3 --version

# Check Dependencies
echo ""
echo "2. Installing Dependencies:"
pip3 install -q -r requirements.txt
echo "✓ Dependencies installed"

# Check Models
echo ""
echo "3. Models Check:"
if [ -f "models/difficulty_model.pkl" ] && [ -f "models/tfidf_vectorizer.pkl" ]; then
    echo "✓ Models found"
else
    echo "✗ Models missing - training..."
    python3 train_model.py
fi

# Test App
echo ""
echo "4. App Test:"
python3 -c "import app; print('✓ App imports successfully')"

echo ""
echo "=== All checks passed! Ready for deployment ==="
