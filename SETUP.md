# Setup Guide

This guide will help you set up the Intelligent Exam Question Analysis project on your local machine.

## Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git (for cloning the repository)
- 4GB RAM minimum
- 1GB free disk space

## Installation Methods

### Method 1: Using pip (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/ArPriCode/INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN.git
cd INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Method 2: Using conda

1. Clone the repository (same as above)

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate exam-question-analysis
```

## Data Setup

1. Ensure your dataset is in the `data/` folder:
```bash
# The file should be at:
data/question_ans_analysis.csv
```

2. If you have the CSV in the root directory, move it:
```bash
mv question_ans_analysis.csv data/
```

## Training the Model

Run the training script:
```bash
python train_model.py
```

Expected output:
```
Loading dataset...
✅ Loaded dataset: 5000 questions

Dataset size: 5000 questions
Difficulty distribution:
...

✅ Models saved successfully!
```

This will create:
- `models/difficulty_model.pkl`
- `models/tfidf_vectorizer.pkl`

## Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Troubleshooting

### Issue: "question_ans_analysis.csv not found"
- Ensure the CSV file is in the `data/` folder
- Check file name spelling (case-sensitive on Linux/macOS)

### Issue: "Module not found"
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### Issue: "Models not found"
- Run `python train_model.py` first
- Check that `models/` folder exists

### Issue: Port 8501 already in use
- Stop other Streamlit apps
- Or use a different port: `streamlit run app.py --server.port 8502`

## Verification

Test your installation:
```bash
python -c "import streamlit, sklearn, pandas, numpy, joblib; print('✅ All dependencies installed!')"
```

## Next Steps

- Read the [README.md](README.md) for project overview
- Check [docs/API.md](docs/API.md) for API documentation
- See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for deployment options
- Review [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) to contribute

## Support

If you encounter issues:
1. Check this troubleshooting section
2. Search existing [GitHub Issues](https://github.com/ArPriCode/INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN/issues)
3. Create a new issue with detailed error messages
