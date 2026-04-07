#!/bin/bash

# Setup script for deployment
echo "Setting up application..."

# Create models directory
mkdir -p models

# Check if models exist
if [ ! -f "models/difficulty_model.pkl" ] || [ ! -f "models/tfidf_vectorizer.pkl" ]; then
    echo "Models not found. Training models..."
    python train_model.py
    echo "Models trained successfully!"
else
    echo "Models already exist. Skipping training."
fi

echo "Setup complete!"

