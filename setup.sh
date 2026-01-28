#!/bin/bash

# ============================================
# Resume Analyzer Setup Script
# ============================================

echo "🚀 Setting up Resume Analyzer..."
echo ""

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
echo "   ✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip
echo "   ✅ Pip upgraded"
echo ""

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt
echo "   ✅ Dependencies installed"
echo ""

# Download spaCy model
echo "🔤 Downloading spaCy language model..."
python -m spacy download en_core_web_sm
echo "   ✅ spaCy model downloaded"
echo ""

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
echo "   ✅ NLTK data downloaded"
echo ""

# Create necessary directories
echo "📁 Creating project directories..."
python -c "from config import ensure_directories; ensure_directories()"
echo "   ✅ Directories created"
echo ""

# Train model for existing dataset
echo "🤖 Checking if ML model needs training..."
if [ -f "data/jobs/job_descriptions.csv" ]; then
    echo "   Training job role prediction model..."
    python ml/train_model.py
    echo "   ✅ Model trained"
else
    echo "   ⚠️  Dataset not found. Model training skipped."
    echo "   Please add job_descriptions.csv to data/jobs/ and run: python ml/train_model.py"
fi
echo ""

# Success message
echo "✅ Setup completed successfully!"
echo ""
echo "🎉 You can now run the application with:"
echo "   streamlit run app.py"
echo ""
echo "📖 For more information, see README.md"
echo ""