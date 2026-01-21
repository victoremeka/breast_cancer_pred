# 🎗️ Breast Cancer Prediction System

An educational machine learning project that classifies breast tumors as **benign** or **malignant** using the Breast Cancer Wisconsin (Diagnostic) Dataset.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org/)

---

## ⚠️ IMPORTANT DISCLAIMER

**This system is strictly for educational purposes and must NOT be used as a medical diagnostic tool.** Always consult qualified healthcare professionals for medical diagnosis and treatment.

---

## 📋 Project Overview

This project implements a complete machine learning pipeline including:
- Data preprocessing and feature selection
- Model training with Logistic Regression
- Model evaluation with comprehensive metrics
- Interactive web application for predictions
- Model persistence for deployment

### Key Features

- ✅ **5 Selected Features**: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean
- ✅ **Algorithm**: Logistic Regression
- ✅ **Preprocessing**: StandardScaler for feature normalization
- ✅ **Web Interface**: User-friendly Streamlit application
- ✅ **High Accuracy**: Well-performing model with excellent metrics

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd breast_cancer_pred
```

2. **Run setup script** (Linux/Mac)
```bash
chmod +x setup.sh
./setup.sh
```

Or **install manually**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Training the Model

Open and run the Jupyter notebook:
```bash
jupyter notebook model/model_building.ipynb
```

Run all cells to generate the model files in the `model/` directory.

### Running the Web Application

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
breast_cancer_pred/
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── setup.sh                            # Setup script
├── README.md                          # Project documentation
├── QUICKSTART.txt                     # Quick start guide
├── BreastCancer_hosted_webGUI_link.txt # Deployment information
├── PROJECT_REQUIREMENTS.md            # Detailed project requirements
│
└── model/
    ├── model_building.ipynb           # Model training notebook
    ├── breast_cancer_model.pkl        # Trained model (generated)
    ├── scaler.pkl                     # Feature scaler (generated)
    └── feature_names.pkl              # Feature names (generated)
```

---

## 🎯 Model Performance

The Logistic Regression model achieves excellent performance metrics:

- **High Accuracy**: Correctly classifies most tumors
- **Good Precision**: Low false positive rate
- **Good Recall**: Low false negative rate
- **Balanced F1-Score**: Good overall performance

*See `model/model_building.ipynb` for detailed metrics and visualizations.*

---

## 💻 Web Application Features

The Streamlit web application provides:

1. **Input Fields**: Easy-to-use number inputs for 5 tumor features
2. **Input Validation**: Ensures valid numeric values
3. **Real-time Predictions**: Instant classification results
4. **Probability Scores**: Confidence levels for predictions
5. **Visual Feedback**: Color-coded results (green for benign, red for malignant)
6. **Educational Disclaimer**: Clear warning about intended use

---

## 🔬 Technical Details

### Dataset

- **Source**: Breast Cancer Wisconsin (Diagnostic) Dataset from sklearn
- **Samples**: 569 tumor samples
- **Classes**: Binary (Benign/Malignant)
- **Split**: 80% training, 20% testing

### Selected Features

1. **mean radius** - Mean of distances from center to points on perimeter
2. **mean texture** - Standard deviation of gray-scale values  
3. **mean perimeter** - Mean size of the core tumor
4. **mean area** - Mean area of the tumor
5. **mean smoothness** - Mean of local variation in radius lengths

### Preprocessing

- **Feature Scaling**: StandardScaler (zero mean, unit variance)
- **Train-Test Split**: Stratified split for balanced classes
- **Random State**: Fixed at 42 for reproducibility

### Model

- **Algorithm**: Logistic Regression
- **Hyperparameters**: max_iter=1000, random_state=42
- **Persistence**: Saved using joblib

---

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy with one click
5. Update `BreastCancer_hosted_webGUI_link.txt` with live URL

### Alternative Platforms

- **Render.com**: Free tier available
- **PythonAnywhere**: Good for Python apps
- **Vercel**: Fast deployment

---

## 📊 Usage Example

```python
# Load the model
import joblib
model = joblib.load('model/breast_cancer_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Prepare input (5 features)
input_data = [[14.5, 19.2, 93.0, 660.0, 0.098]]

# Scale and predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)

print(f"Prediction: {'Benign' if prediction[0] == 1 else 'Malignant'}")
print(f"Confidence: {max(probability[0]):.2%}")
```

---

## 🛠️ Development

### Running Tests

```bash
# Test the web app
streamlit run app.py

# Test model loading
python -c "import joblib; print(joblib.load('model/breast_cancer_model.pkl'))"
```

### Adding New Features

To modify the selected features:
1. Update the feature selection in `model/model_building.ipynb`
2. Retrain the model
3. Update the input fields in `app.py`
4. Update this README

---

## 📝 Assignment Information

- **Student**: Victor Emeka
- **Matric Number**: 23cg034065
- **Course**: Machine Learning / AI
- **Submission**: Scorac.com
- **Deadline**: Friday, January 22, 2026 at 11:59 PM

---

## 🤝 Contributing

This is an educational project for academic purposes. For improvements or issues:
1. Document the change needed
2. Test thoroughly
3. Update documentation

---

## 📚 Resources

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Breast Cancer Dataset Info](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- [Logistic Regression Guide](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

## 🐛 Troubleshooting

### Model files not found
- Run the Jupyter notebook first to generate model files
- Check that files exist in `model/` directory

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

### Streamlit issues
- Check port availability: Try `streamlit run app.py --server.port 8502`
- Clear cache: `streamlit cache clear`

---

## 📄 License

This is an educational project. All rights reserved.

---

## 👥 Contact

For questions or support:
- Contact your course instructor
- Review the PROJECT_REQUIREMENTS.md file
- Check the QUICKSTART.txt guide

---

**Last Updated**: January 21, 2026

**Built with** ❤️ **for educational purposes**
