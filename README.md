# CodeAlpha Credit Scoring Model

## Overview

This project was developed as part of the CodeAlpha Machine Learning Internship Program.

The objective is to predict an individual's creditworthiness using historical financial information and machine learning algorithms.

## Problem Statement

Financial institutions need accurate methods to assess whether a loan applicant is likely to repay a loan. This project automates credit risk assessment using machine learning classification models.

## Dataset

Credit Risk Dataset (Kaggle)

Dataset Size:

- 32,581 records
- 12 features

Features:

- Person Age
- Income
- Employment Length
- Home Ownership
- Loan Intent
- Loan Grade
- Loan Amount
- Interest Rate
- Loan Percent Income
- Previous Default History
- Credit History Length

Target Variable:

- loan_status

## Machine Learning Models

The following classification models were evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Model Comparison

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 82.91%   |
| Decision Tree       | 88.55%   |
| Random Forest       | 92.93%   |

Random Forest achieved the highest accuracy and was selected as the final model.

## Performance Metrics (Random Forest)

Accuracy: 92.93%

Precision: 97%

Recall: 71%

F1-Score: 82%

ROC-AUC: 0.849

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Visualizations

- Feature Importance Analysis
- Confusion Matrix
- ROC Curve
- Model Comparison Chart

## Streamlit Application

The application allows users to:

- Enter applicant information
- Predict credit risk
- View risk probability
- Categorize applicants into:
  - Low Risk
  - Medium Risk
  - High Risk

## Installation

pip install -r requirements.txt

python train.py

streamlit run app.py

## Future Improvements

- Hyperparameter Tuning
- XGBoost Integration
- SHAP Explainability
- Cloud Deployment

## Author

Sukhraj Singh

CodeAlpha Machine Learning Internship
