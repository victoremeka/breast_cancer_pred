"""
Breast Cancer Prediction System - Web Application
Educational ML Project for Tumor Classification

⚠️ DISCLAIMER: This system is strictly for educational purposes 
and must NOT be used as a medical diagnostic tool.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🎗️",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #FF1493;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .disclaimer {
        background-color: #FFF3CD;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FFC107;
        margin-bottom: 20px;
    }
    .result-benign {
        background-color: #D4EDDA;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28A745;
        margin-top: 20px;
    }
    .result-malignant {
        background-color: #F8D7DA;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #DC3545;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model, scaler, and feature names
@st.cache_resource
def load_model_artifacts():
    """Load the trained model, scaler, and feature names"""
    try:
        model_path = os.path.join('model', 'breast_cancer_model.pkl')
        scaler_path = os.path.join('model', 'scaler.pkl')
        features_path = os.path.join('model', 'feature_names.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("⚠️ Error: Model files not found. Please run the model_building.ipynb notebook first.")
        st.stop()

# Load artifacts
model, scaler, feature_names = load_model_artifacts()

# Header
st.markdown('<div class="main-header">🎗️ Breast Cancer Prediction System</div>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ IMPORTANT DISCLAIMER</strong><br>
    This system is strictly for <strong>educational purposes</strong> and must <strong>NOT</strong> 
    be used as a medical diagnostic tool. Always consult qualified healthcare professionals 
    for medical diagnosis and treatment.
</div>
""", unsafe_allow_html=True)

# Information section
with st.expander("ℹ️ About This System"):
    st.write("""
    This machine learning system classifies breast tumors as **benign** or **malignant** 
    based on the Breast Cancer Wisconsin (Diagnostic) Dataset.
    
    **Algorithm**: Logistic Regression  
    **Features Used**: 5 tumor characteristics
    
    **Model Performance**:
    - High accuracy and precision
    - Trained on validated medical dataset
    - Implements StandardScaler preprocessing
    """)

# Input section
st.markdown("### 📊 Enter Tumor Feature Values")
st.markdown("Please enter the following measurements for the tumor:")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input(
        "Mean Radius",
        min_value=0.0,
        max_value=50.0,
        value=14.0,
        step=0.1,
        help="Mean of distances from center to points on the perimeter"
    )
    
    texture_mean = st.number_input(
        "Mean Texture",
        min_value=0.0,
        max_value=50.0,
        value=19.0,
        step=0.1,
        help="Standard deviation of gray-scale values"
    )
    
    perimeter_mean = st.number_input(
        "Mean Perimeter",
        min_value=0.0,
        max_value=200.0,
        value=92.0,
        step=0.1,
        help="Mean size of the core tumor perimeter"
    )

with col2:
    area_mean = st.number_input(
        "Mean Area",
        min_value=0.0,
        max_value=2500.0,
        value=655.0,
        step=1.0,
        help="Mean area of the tumor"
    )
    
    smoothness_mean = st.number_input(
        "Mean Smoothness",
        min_value=0.0,
        max_value=0.5,
        value=0.096,
        step=0.001,
        format="%.3f",
        help="Mean of local variation in radius lengths"
    )

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Predict button
if st.button("🔍 Predict Diagnosis", type="primary", use_container_width=True):
    try:
        # Validate inputs
        if any(val < 0 for val in [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]):
            st.error("⚠️ Error: All values must be non-negative numbers.")
        else:
            # Prepare input data
            input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])
            
            # Create DataFrame with correct feature names
            input_df = pd.DataFrame(input_data, columns=feature_names)
            
            # Scale the input using the saved scaler
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            
            if prediction == 1:  # Benign
                st.markdown("""
                <div class="result-benign">
                    <h2 style="color: #28A745; margin-top: 0;">✅ Prediction: BENIGN</h2>
                    <p style="font-size: 1.2em;">The tumor is predicted to be <strong>benign (non-cancerous)</strong>.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.success(f"**Confidence**: {probability[1]:.2%}")
                
            else:  # Malignant
                st.markdown("""
                <div class="result-malignant">
                    <h2 style="color: #DC3545; margin-top: 0;">⚠️ Prediction: MALIGNANT</h2>
                    <p style="font-size: 1.2em;">The tumor is predicted to be <strong>malignant (cancerous)</strong>.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.error(f"**Confidence**: {probability[0]:.2%}")
            
            # Show probability breakdown
            st.markdown("### 📈 Prediction Probabilities")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Malignant Probability",
                    value=f"{probability[0]:.2%}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Benign Probability", 
                    value=f"{probability[1]:.2%}",
                    delta=None
                )
            
            # Reminder
            st.info("💡 **Reminder**: This is an educational tool. Please consult a medical professional for actual diagnosis.")
            
    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {str(e)}")
        st.write("Please check your input values and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>Breast Cancer Prediction System | Educational ML Project</p>
    <p>Built with Streamlit & Scikit-learn | Logistic Regression Model</p>
</div>
""", unsafe_allow_html=True)
