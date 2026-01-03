# Breast Cancer Prediction using Machine Learning

This project explores the use of machine learning models to predict breast cancer (benign vs malignant) using medical diagnostic data. The goal is to evaluate different classification models and understand which features contribute most to accurate early detection.

This work is based on the Breast Cancer Wisconsin (Diagnostic) dataset and was developed as part of a Machine Learning course project.

---

## Overview

Early detection of breast cancer is critical for improving patient outcomes. Traditional diagnostic methods can be invasive and expensive. This project applies supervised machine learning techniques to assist in early diagnosis by identifying patterns in digitized medical data.

Multiple models are trained and compared to evaluate accuracy, reliability, and interpretability.

---

## Models Used

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  

---

## Dataset

- Source: UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic)
- Features extracted from digitized images of fine needle aspirates (FNA) of breast masses
- Target variable: Benign vs Malignant classification

---

## Approach

1. **Data Preprocessing**
   - Feature scaling
   - Handling missing values
   - Addressing class imbalance

2. **Model Training & Evaluation**
   - Train/test split
   - Accuracy, confusion matrix, and classification reports

3. **Analysis**
   - Model performance comparison
   - Feature importance analysis (Random Forest)
   - Error analysis with emphasis on minimizing false negatives

---

## Results & Insights

- Logistic Regression and SVM achieved strong predictive performance
- Decision Tree models showed signs of overfitting
- Feature importance analysis highlighted key medical indicators relevant for diagnosis
- Results reinforce the value of machine learning in supporting early cancer detection

---

## Tech Stack

- Python
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- Jupyter Notebook

---

## Repository Structure

```text
.
├── Cancer_Prediction.ipynb
├── ML_Report.pdf
└── README.md
