# Quick Start Guide

Get up and running in 5 minutes!

##  Fast Setup

```bash
# 1. Clone repository
git clone https://github.com/ArPriCode/INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN.git
cd INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify setup
python scripts/verify_setup.py

# 4. Train model (if needed)
python train_model.py

# 5. Run application
streamlit run app.py
```

##  Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset in `data/` folder
- [ ] Models trained (`python train_model.py`)
- [ ] App running (`streamlit run app.py`)

## 🔧 Common Commands

```bash
# Verify setup
python scripts/verify_setup.py

# Train models
python train_model.py

# Run app
streamlit run app.py

# Run app on different port
streamlit run app.py --server.port 8502

# Check dependencies
pip list | grep -E "streamlit|scikit-learn|pandas"
```

##  Next Steps

- Read [README.md](README.md) for full documentation
- Check [SETUP.md](SETUP.md) for detailed setup
- See [docs/](docs/) for API and deployment guides

## 🆘 Need Help?

- Run verification: `python scripts/verify_setup.py`
- Check [SETUP.md](SETUP.md) troubleshooting section
- Open an issue on GitHub
