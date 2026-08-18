# IBM HR Analytics Employee Attrition Classification

## Problem Statement
Build and compare classification models to predict employee attrition.

## Dataset
IBM HR Analytics Employee Attrition & Performance dataset from Kaggle. It contains 1,470 records and 35 original columns. After removing the target and non-predictive identifier/constant fields, 30 predictive features remain.

Kaggle: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Github repository - https://github.com/2025ac05932-arch/peoject-folder
## Models
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Ensemble

The assignment text says six models but explicitly lists five; this implementation follows the five listed models.

## Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC).

## Run Streamlit
```bash
pip install -r requirements.txt
streamlit run app.py
```

##Notebook name - 2025ac05932_ML_assignment.ipynb
