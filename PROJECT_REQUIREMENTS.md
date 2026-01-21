# Breast Cancer Prediction System - Project Requirements

## 📋 Project Overview

Develop a **Breast Cancer Prediction System** using machine learning to classify tumors as **benign** or **malignant** based on features from the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

### ⚠️ Important Disclaimer
**This system is strictly for educational purposes and must NOT be presented as a medical diagnostic tool.**

---

## 📊 Dataset Information

**Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI / sklearn)

### Available Features

Select **any FIVE (5)** input features from the following:

- `radius_mean` - Mean of distances from center to points on perimeter
- `texture_mean` - Standard deviation of gray-scale values
- `perimeter_mean` - Mean size of the core tumor
- `area_mean` - Mean area of the tumor
- `smoothness_mean` - Mean of local variation in radius lengths
- `compactness_mean` - Mean of (perimeter² / area - 1.0)
- `concavity_mean` - Mean of severity of concave portions of contour
- `symmetry_mean` - Mean symmetry of the tumor

### Target Variable (Output)

- `diagnosis` - Binary classification: **Benign (B)** or **Malignant (M)**

---

## 📝 PART A — Model Development

**File**: `model_building.ipynb` or `model_development.py`

### Requirements

1. **Load Dataset**
   - Use the Breast Cancer Wisconsin dataset from sklearn or UCI

2. **Data Preprocessing**
   - Handle missing values (if any)
   - Select **exactly 5 features** from the recommended list
   - Encode the target variable (`diagnosis`: B → 0, M → 1)
   - **Mandatory**: Apply feature scaling (StandardScaler or MinMaxScaler)
     - *Critical for distance-based algorithms (SVM, KNN, Neural Networks)*

3. **Train-Test Split**
   - Split data into training and testing sets (e.g., 80-20 or 70-30)
   - Use `random_state` for reproducibility

4. **Model Selection**
   
   Implement **ONE** of the following algorithms:
   - ✅ Logistic Regression
   - ✅ Support Vector Machine (SVM)
   - ✅ K-Nearest Neighbors (KNN)
   - ✅ Naïve Bayes
   - ✅ Neural Networks (MLP Classifier)
   - ✅ LightGBM

5. **Model Training**
   - Train the model on the training dataset
   - Tune hyperparameters (optional but recommended)

6. **Model Evaluation**
   
   Calculate and display the following metrics:
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1-Score**
   - **Confusion Matrix** (optional but recommended)
   - **Classification Report**

7. **Model Persistence**
   - Save the trained model using **Joblib** or **Pickle**
   - Save the scaler separately (important for preprocessing new data)
   - **Demonstrate** that the saved model can be reloaded and make predictions

### Example Code Structure

```python
# Save model and scaler
import joblib
joblib.dump(model, 'model/breast_cancer_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Load and test
loaded_model = joblib.load('model/breast_cancer_model.pkl')
loaded_scaler = joblib.load('model/scaler.pkl')
# Make test prediction...
```

---

## 🌐 PART B — Web GUI Application

**Files**: `app.py` and `index.html` (if applicable)

### Requirements

Develop a web-based interface that:

1. **Loads the saved model** and scaler from disk
2. **Provides input fields** for the 5 selected tumor features
3. **Validates user input** (ensures numeric values within reasonable ranges)
4. **Preprocesses input** using the saved scaler
5. **Makes predictions** using the loaded model
6. **Displays results** clearly:
   - Prediction: Benign or Malignant
   - Confidence/Probability (optional)
   - Visual indicators (colors, icons)

### Technology Stack (Choose ONE)

| Framework | Difficulty | Best For |
|-----------|-----------|----------|
| **Streamlit** | ⭐ Easy | Quick prototypes, interactive dashboards |
| **Flask** + HTML/CSS | ⭐⭐ Medium | Custom UI, full control |
| **Gradio** | ⭐ Easy | ML demos, simple interfaces |
| **FastAPI** | ⭐⭐⭐ Advanced | API-first, production apps |
| **Django** | ⭐⭐⭐ Advanced | Not recommended for this project |

### Design Guidelines

- Clear, user-friendly interface
- Input validation with helpful error messages
- Professional appearance
- Responsive design (optional)
- Include disclaimer about educational use

---

## 📁 PART C — GitHub Submission

### Required Project Structure

```
/BreastCancer_Project_yourName_matricNo/
│
├── app.py                              # Main web application
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── BreastCancer_hosted_webGUI_link.txt # Deployment info
│
├── model/
│   ├── model_building.ipynb           # Jupyter notebook with full workflow
│   ├── breast_cancer_model.pkl        # Trained model
│   └── scaler.pkl                     # Fitted scaler
│
├── static/                            # (Optional) CSS, images
│   └── style.css
│
└── templates/                         # (Optional) HTML templates for Flask
    └── index.html
```

### Git Workflow

```bash
git init
git add .
git commit -m "Initial commit: Breast Cancer Prediction System"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

---

## 🚀 PART D — Deployment

### Deployment Platforms (Choose ONE)

1. **Streamlit Cloud** ⭐ Recommended
   - Free tier available
   - Excellent for Streamlit apps
   - Auto-deploys from GitHub
   - URL: https://streamlit.io/cloud

2. **Render.com**
   - Free tier available
   - Supports Flask, FastAPI
   - URL: https://render.com

3. **PythonAnywhere**
   - Free tier available
   - Good for Flask apps
   - URL: https://www.pythonanywhere.com

4. **Vercel**
   - Free tier available
   - Fast deployment
   - URL: https://vercel.com

### Deployment Checklist

- [ ] Create `requirements.txt` with all dependencies
- [ ] Add `.gitignore` (exclude unnecessary files)
- [ ] Ensure model files (.pkl) are in the repository
- [ ] Test application locally before deploying
- [ ] Configure environment variables (if needed)
- [ ] Deploy and test the live application
- [ ] Verify all features work in production

---

## 📤 Scorac.com Submission Requirements

### Submission Deadline
**Friday, January 22, 2026 at 11:59 PM**

### File: `BreastCancer_hosted_webGUI_link.txt`

Create this file with the following information:

```
Name: [Your Full Name]
Matric Number: [Your Matric Number]
Machine Learning Algorithm Used: [e.g., Logistic Regression]
Model Persistence Method: [Joblib or Pickle]
Live Application URL: [Your deployed app URL]
GitHub Repository: [Your GitHub repo URL]
Features Selected: [List your 5 selected features]

Additional Notes:
[Any important information about your implementation]
```

### Final Submission Structure

Upload to Scorac.com with this exact structure:

```
/BreastCancer_Project_yourName_matricNo/
├── app.py
├── requirements.txt
├── BreastCancer_hosted_webGUI_link.txt
├── README.md                          # Optional but recommended
│
├── model/
│   ├── model_building.ipynb
│   ├── breast_cancer_model.pkl
│   └── scaler.pkl
│
├── static/                            # If applicable
│   └── style.css
│
└── templates/                         # If applicable
    └── index.html
```

---

## ✅ Evaluation Criteria

Your project will be evaluated on:

1. **Model Performance** (30%)
   - Appropriate algorithm selection
   - Proper preprocessing and scaling
   - Good evaluation metrics (accuracy, precision, recall, F1)

2. **Code Quality** (25%)
   - Clean, well-organized code
   - Proper comments and documentation
   - Follows best practices

3. **Web Application** (25%)
   - Functional and user-friendly interface
   - Proper input validation
   - Clear result display
   - Successfully deployed

4. **Documentation** (10%)
   - Clear README
   - Proper project structure
   - Complete submission file

5. **GitHub Repository** (10%)
   - Well-organized structure
   - Clear commit history
   - Complete files

---

## 🛠️ Helpful Resources

### Documentation
- [scikit-learn Breast Cancer Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Joblib Documentation](https://joblib.readthedocs.io/)

### Example Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Run Flask app
python app.py
```

---

## ⚠️ Common Pitfalls to Avoid

1. ❌ Forgetting to scale input features during prediction
2. ❌ Not saving the scaler along with the model
3. ❌ Using different feature sets for training and prediction
4. ❌ Not handling invalid user input
5. ❌ Model files not pushed to GitHub
6. ❌ Incorrect file paths in deployed application
7. ❌ Missing dependencies in requirements.txt
8. ❌ Not including the medical disclaimer

---

## 💡 Tips for Success

✅ **Start early** - Don't wait until the deadline
✅ **Test locally first** - Ensure everything works before deploying
✅ **Use version control** - Commit frequently with meaningful messages
✅ **Document your work** - Add comments and create a good README
✅ **Handle errors gracefully** - Add try-except blocks and user-friendly messages
✅ **Keep it simple** - Focus on functionality over fancy features
✅ **Ask for help** - If stuck, reach out to instructors or classmates

---

## 📞 Support

If you encounter issues:
1. Check the error messages carefully
2. Review the documentation
3. Search for similar issues online
4. Ask your instructor or TA
5. Collaborate with classmates (but submit individual work)

---

**Good luck with your project! 🚀**

*Last Updated: January 21, 2026*
