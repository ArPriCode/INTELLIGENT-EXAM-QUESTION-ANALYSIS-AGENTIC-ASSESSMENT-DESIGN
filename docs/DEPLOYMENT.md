#  Deployment Guide

Complete guide for deploying the Exam Question Analysis system to various platforms.

---

##  Pre-Deployment Checklist

- [ ] All dependencies listed in `requirements.txt`
- [ ] Models trained and saved in `models/` directory
- [ ] `.gitignore` configured properly
- [ ] Environment variables documented
- [ ] README.md updated
- [ ] Code tested locally

---

## 1⃣ Streamlit Community Cloud

### Advantages
-  Free hosting
-  Automatic deployments from GitHub
-  Built-in SSL
-  Easy setup

### Steps

1. **Prepare Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial deployment"
   git remote add origin https://github.com/ArPriCode/INTELLIGENT-EXAM-QUESTION-ANALYSIS-AGENTIC-ASSESSMENT-DESIGN.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect GitHub account
   - Select repository and branch
   - Set main file: `app.py`
   - Click "Deploy"

3. **Configure Secrets** (if needed)
   - Go to App settings → Secrets
   - Add environment variables in TOML format

4. **Custom Domain** (Optional)
   - Go to App settings → General
   - Add custom domain

### Troubleshooting

**Issue**: Models not found
```bash
# Solution: Add setup.sh
#!/bin/bash
python train_model.py
```

**Issue**: Memory limit exceeded
```bash
# Solution: Optimize model size or upgrade plan
```

---

## 2⃣ Hugging Face Spaces

### Advantages
-  Free GPU access
-  ML-focused community
-  Easy model sharing
-  Version control

### Steps

1. **Create Space**
   - Visit [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Select "Streamlit" as SDK
   - Choose visibility (Public/Private)

2. **Clone Space Repository**
   ```bash
   git clone https://huggingface.co/spaces/ArPriCode/exam-question-analysis
   cd exam-question-analysis
   ```

3. **Add Files**
   ```bash
   cp -r /path/to/your/project/* .
   git add .
   git commit -m "Deploy to Hugging Face"
   git push
   ```

4. **Configure Space**
   Create `README.md` with metadata:
   ```yaml
   ---
   title: Exam Question Analysis
   emoji: 
   colorFrom: blue
   colorTo: purple
   sdk: streamlit
   sdk_version: 1.28.0
   app_file: app.py
   pinned: false
   ---
   ```

5. **Add Requirements**
   Ensure `requirements.txt` includes all dependencies

### Environment Variables

Create `.env` file (add to `.gitignore`):
```bash
API_KEY=your_api_key_here
MODEL_PATH=models/
```

---

## 3⃣ Render

### Advantages
-  Free tier available
-  Automatic HTTPS
-  Custom domains
-  Database support

### Steps

1. **Create `render.yaml`**
   ```yaml
   services:
     - type: web
       name: exam-question-analysis
       env: python
       region: oregon
       plan: free
       buildCommand: |
         pip install -r requirements.txt
         python train_model.py
       startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
       envVars:
         - key: PYTHON_VERSION
           value: 3.11.0
   ```

2. **Deploy**
   - Visit [render.com](https://render.com)
   - Click "New +" → "Blueprint"
   - Connect GitHub repository
   - Render auto-detects `render.yaml`
   - Click "Apply"

3. **Configure Environment**
   - Go to Dashboard → Environment
   - Add environment variables

### Custom Domain

1. Go to Settings → Custom Domain
2. Add your domain
3. Update DNS records:
   ```
   Type: CNAME
   Name: www
   Value: your-app.onrender.com
   ```

---

## 4⃣ Railway

### Advantages
-  Simple deployment
-  Free tier
-  Database integration
-  Auto-scaling

### Steps

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login**
   ```bash
   railway login
   ```

3. **Initialize Project**
   ```bash
   railway init
   ```

4. **Deploy**
   ```bash
   railway up
   ```

5. **Configure Start Command**
   ```bash
   railway run streamlit run app.py
   ```

---

## 5⃣ Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Train model
RUN python train_model.py

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t exam-analysis .

# Run container
docker run -p 8501:8501 exam-analysis
```

### Docker Compose

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

---

## 6⃣ AWS EC2

### Steps

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04
   - Instance type: t2.micro (free tier)
   - Security group: Allow port 8501

2. **Connect to Instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip
   pip3 install -r requirements.txt
   ```

4. **Run Application**
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

5. **Setup as Service**
   Create `/etc/systemd/system/streamlit.service`:
   ```ini
   [Unit]
   Description=Streamlit App
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/app
   ExecStart=/usr/local/bin/streamlit run app.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Enable service:
   ```bash
   sudo systemctl enable streamlit
   sudo systemctl start streamlit
   ```

---

##  Security Best Practices

### 1. Environment Variables
Never commit sensitive data. Use environment variables:

```python
import os
API_KEY = os.getenv('API_KEY')
```

### 2. HTTPS
Always use HTTPS in production. Most platforms provide it automatically.

### 3. Rate Limiting
Implement rate limiting to prevent abuse:

```python
import streamlit as st
from datetime import datetime, timedelta

if 'last_request' not in st.session_state:
    st.session_state.last_request = datetime.now()

if datetime.now() - st.session_state.last_request < timedelta(seconds=1):
    st.error("Please wait before making another request")
else:
    st.session_state.last_request = datetime.now()
    # Process request
```

### 4. Input Validation
Validate all user inputs:

```python
def validate_question(text):
    if not text or len(text) < 10:
        return False, "Question too short"
    if len(text) > 1000:
        return False, "Question too long"
    return True, "Valid"
```

---

##  Monitoring

### Streamlit Analytics

```python
import streamlit as st

# Track usage
if 'page_views' not in st.session_state:
    st.session_state.page_views = 0
st.session_state.page_views += 1
```

### Error Logging

```python
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    # Your code
    pass
except Exception as e:
    logging.error(f"Error: {str(e)}")
```

---

##  CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          python -m pytest tests/
      
      - name: Train model
        run: |
          python train_model.py
```

---

##  Post-Deployment

### 1. Test Application
- [ ] All pages load correctly
- [ ] Model predictions work
- [ ] File uploads function
- [ ] Downloads work
- [ ] Mobile responsive

### 2. Monitor Performance
- [ ] Response times
- [ ] Error rates
- [ ] User analytics
- [ ] Resource usage

### 3. Update Documentation
- [ ] Add deployment URL to README
- [ ] Document any issues
- [ ] Update changelog

---

##  Troubleshooting

### Common Issues

**Issue**: Port already in use
```bash
# Solution: Kill process on port 8501
lsof -ti:8501 | xargs kill -9
```

**Issue**: Module not found
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Issue**: Model file not found
```bash
# Solution: Retrain model
python train_model.py
```

---

##  Support

For deployment issues:
- Check platform documentation
- Review application logs
- Contact platform support
- Open GitHub issue

---

**Happy Deploying! **
