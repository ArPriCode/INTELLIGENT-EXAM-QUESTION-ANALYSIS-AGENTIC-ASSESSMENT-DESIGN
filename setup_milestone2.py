#!/usr/bin/env python3
"""
Setup script for Milestone 2 dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Installing {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_dependencies():
    """Install Milestone 2 dependencies"""
    print("Installing Milestone 2 Dependencies")
    print("=" * 50)
    
    # Core LangGraph dependencies
    dependencies = [
        "langgraph>=0.0.1",
        "langchain>=0.1.0", 
        "langchain-community>=0.0.1",
        "langchain-openai>=0.0.1",
        "chromadb>=0.4.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠ Failed to install {dep}, continuing...")
    
    # Optional dependencies
    print("\nInstalling optional dependencies...")
    optional_deps = [
        "ollama>=0.1.0",
        "faiss-cpu>=1.7.4"
    ]
    
    for dep in optional_deps:
        run_command(f"pip install {dep}", f"Installing {dep} (optional)")

def setup_environment():
    """Setup environment files"""
    print("\nSetting up environment...")
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        env_content = """# Milestone 2 Configuration
# Ollama Configuration (for local LLM)
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI Configuration (optional)
# OPENAI_API_KEY=your_api_key_here

# Chroma Database Path
CHROMA_DB_PATH=./chroma_db

# Python Configuration
PYTHONUNBUFFERED=1
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✓ Created .env file")
    else:
        print("✓ .env file already exists")

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = ['chroma_db', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created {directory}/ directory")
        else:
            print(f"✓ {directory}/ directory already exists")

def test_installation():
    """Test if installation was successful"""
    print("\nTesting installation...")
    
    try:
        from langgraph.graph import StateGraph, END
        from langchain_core.messages import HumanMessage
        import chromadb
        print("✓ All core dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 50)
    print("MILESTONE 2 SETUP COMPLETE!")
    print("=" * 50)
    
    print("\nNext Steps:")
    print("1. Run Milestone 1 app:")
    print("   streamlit run app.py")
    
    print("\n2. Run Milestone 2 app:")
    print("   streamlit run app_milestone2.py")
    
    print("\n3. Optional: Install Ollama for local LLM")
    print("   - macOS: brew install ollama")
    print("   - Linux: curl https://ollama.ai/install.sh | sh")
    print("   - Then: ollama pull mistral")
    
    print("\n4. Optional: Add OpenAI API key to .env file")
    print("   OPENAI_API_KEY=your_key_here")
    
    print("\nDocumentation:")
    print("   - README.md: Complete project documentation")
    print("   - docs/MILESTONE2.md: Milestone 2 specific docs")

def main():
    """Main setup function"""
    print("Milestone 2 Setup Script")
    print("This will install LangGraph and agentic AI dependencies")
    print("=" * 60)
    
    # Install dependencies
    install_dependencies()
    
    # Setup environment
    setup_environment()
    
    # Create directories
    create_directories()
    
    # Test installation
    if test_installation():
        print_next_steps()
    else:
        print("\n✗ Installation test failed. Please check error messages above.")
        print("You may need to restart your Python environment or install dependencies manually.")

if __name__ == "__main__":
    main()