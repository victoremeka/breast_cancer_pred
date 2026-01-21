#!/bin/bash

echo "Setting up Breast Cancer Prediction System..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Run 'streamlit run app.py' to start the application."
