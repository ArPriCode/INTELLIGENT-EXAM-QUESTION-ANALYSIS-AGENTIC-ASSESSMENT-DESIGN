#  Contributing to Exam Question Analysis

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

---

##  Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

---

##  Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

---

##  Getting Started

### Prerequisites

- Python 3.8+
- Git
- GitHub account
- Basic knowledge of ML/NLP
- Familiarity with Streamlit (optional)

### First Time Contributors

1. **Find an Issue**
   - Look for issues labeled `good first issue`
   - Comment on the issue to claim it
   - Wait for maintainer approval

2. **Ask Questions**
   - Don't hesitate to ask for clarification
   - Use GitHub Discussions for general questions
   - Tag maintainers if needed

---

##  Development Setup

### 1. Fork the Repository

Click the "Fork" button on GitHub to create your copy.

### 2. Clone Your Fork

```bash
git clone https://github.com/ArPriCode/INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN.git
cd INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN
```

### 3. Add Upstream Remote

```bash
git remote add upstream https://github.com/ArPriCode/INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN.git
```

### 4. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 6. Train Model

```bash
python train_model.py
```

### 7. Run Application

```bash
streamlit run app.py
```

---

##  How to Contribute

### Types of Contributions

1. **Bug Fixes**
   - Fix existing bugs
   - Add tests for the fix
   - Update documentation if needed

2. **New Features**
   - Discuss feature in an issue first
   - Implement feature
   - Add tests and documentation

3. **Documentation**
   - Improve README
   - Add code comments
   - Create tutorials

4. **Testing**
   - Add unit tests
   - Improve test coverage
   - Add integration tests

### Contribution Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make Changes**
   - Write clean, readable code
   - Follow coding standards
   - Add comments where necessary

3. **Test Your Changes**
   ```bash
   python -m pytest tests/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Go to GitHub
   - Click "New Pull Request"
   - Fill in the template
   - Link related issues

---

##  Coding Standards

### Python Style Guide

Follow **PEP 8** guidelines:

```python
# Good
def predict_difficulty(question_text, model, vectorizer):
    """
    Predict difficulty level of a question.
    
    Args:
        question_text (str): The exam question text
        model: Trained classifier model
        vectorizer: Fitted TF-IDF vectorizer
    
    Returns:
        tuple: (prediction, probability_array)
    """
    features = vectorizer.transform([question_text])
    prediction = model.predict(features)[0]
    return prediction, model.predict_proba(features)[0]

# Bad
def pred(q,m,v):
    f=v.transform([q])
    p=m.predict(f)[0]
    return p,m.predict_proba(f)[0]
```

### Code Formatting

Use **Black** for formatting:

```bash
pip install black
black app.py train_model.py
```

### Import Order

```python
# Standard library imports
import os
import sys

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Local imports
from utils import preprocess_text
```

### Naming Conventions

```python
# Variables and functions: snake_case
user_input = "example"
def calculate_metrics():
    pass

# Classes: PascalCase
class DifficultyPredictor:
    pass

# Constants: UPPER_CASE
MAX_WORD_COUNT = 1000
DEFAULT_MODEL_PATH = "models/"
```

### Documentation

Add docstrings to all functions:

```python
def train_model(data_path, test_size=0.2):
    """
    Train difficulty prediction model.
    
    Args:
        data_path (str): Path to training data CSV
        test_size (float): Proportion of test set (default: 0.2)
    
    Returns:
        tuple: (model, vectorizer, metrics)
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If test_size not in (0, 1)
    
    Example:
        >>> model, vec, metrics = train_model('data.csv')
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    pass
```

---

##  Testing

### Writing Tests

Create tests in `tests/` directory:

```python
# tests/test_model.py
import pytest
from train_model import predict_difficulty, load_models

def test_predict_difficulty():
    """Test difficulty prediction"""
    model, vectorizer = load_models()
    question = "What is 2+2?"
    pred, prob = predict_difficulty(question, model, vectorizer)
    
    assert pred in [0, 1, 2]
    assert len(prob) == 3
    assert sum(prob) == pytest.approx(1.0)

def test_invalid_input():
    """Test handling of invalid input"""
    model, vectorizer = load_models()
    pred, prob = predict_difficulty("", model, vectorizer)
    
    assert pred is None
    assert prob is None
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Test Coverage

Aim for >80% code coverage:

```bash
pip install pytest-cov
pytest --cov=. --cov-report=term-missing
```

---

##  Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe testing performed

## Screenshots (if applicable)
Add screenshots

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
```

### Review Process

1. **Automated Checks**
   - CI/CD pipeline runs
   - Tests must pass
   - Code quality checks

2. **Code Review**
   - At least one maintainer reviews
   - Address feedback
   - Make requested changes

3. **Approval**
   - Maintainer approves PR
   - PR is merged to main

---

##  Issue Guidelines

### Creating Issues

Use appropriate templates:

#### Bug Report

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What should happen

**Screenshots**
If applicable

**Environment**
- OS: [e.g., macOS]
- Python version: [e.g., 3.11]
- Browser: [e.g., Chrome]
```

#### Feature Request

```markdown
**Is your feature request related to a problem?**
Description of problem

**Describe the solution**
Proposed solution

**Alternatives considered**
Other approaches

**Additional context**
Any other information
```

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information requested

---

##  Development Priorities

### High Priority
- Bug fixes
- Security improvements
- Performance optimization
- Critical features

### Medium Priority
- New features
- UI improvements
- Documentation
- Test coverage

### Low Priority
- Code refactoring
- Minor enhancements
- Nice-to-have features

---

##  Resources

### Learning Resources
- [Python PEP 8](https://pep8.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Git Workflow](https://guides.github.com/introduction/flow/)

### Tools
- [Black](https://black.readthedocs.io/) - Code formatter
- [Pylint](https://pylint.org/) - Code linter
- [pytest](https://pytest.org/) - Testing framework

---

##  Communication

### Channels
- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code contributions

### Response Times
- Issues: Within 48 hours
- Pull Requests: Within 1 week
- Questions: Within 24 hours

---

##  Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Credited in documentation

---

##  Contact

- **Project Lead**: ArPriCode Team
- **Email**: contact@arpricode.dev
- **GitHub**: [@ArPriCode](https://github.com/ArPriCode)

---

##  License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing! **
