# IBM HR Analytics Employee Attrition Classification

## Problem Statement
Build and compare classification models to predict employee attrition.

## Dataset
IBM HR Analytics Employee Attrition & Performance dataset from Kaggle. It contains 1,470 records and 35 original columns. After removing the target and non-predictive identifier/constant fields, 30 predictive features remain.

Kaggle: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Github repository - https://github.com/2025ac05932-arch/project-folder
## Models
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Ensemble

The assignment text says six models but explicitly lists five; this implementation follows the five listed models.

## Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC).

Models comparison 

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.7517	0.8032	0.3488	0.6383	0.4511	0.3316
Decision Tree	0.7857	0.6527	0.3788	0.5319	0.4425	0.3214
KNN	0.8401	0.6144	0.5	0.1064	0.1754	0.1742
Naive Bayes	0.6463	0.7032	0.2605	0.6596	0.3735	0.2265
Random Forest	0.8435	0.7893	0.5217	0.2553	0.3429	0.2877


ML Model Name	Observation about model performance
Logistic Regression
Provides a good balance between AUC (0.8032) and Recall (0.6383). It identifies a relatively high number of employees likely to leave, although its Accuracy (0.7517) is lower than the tree-based models.
Decision Tree	
Achieves reasonably good Accuracy (0.7857), but its AUC (0.6527) is relatively low. It has moderate Recall (0.5319) and F1 (0.4425), indicating average performance in identifying employees likely to leave.
KNN	
Has high Accuracy (0.8401) but very low Recall (0.1064) and F1 (0.1754). This suggests that it predicts the majority class well but performs poorly at identifying employees who actually leave.
Naive Bayes	
Has the lowest Accuracy (0.6463), but relatively high Recall (0.6596). It identifies many potential attrition cases but generates more false positives, resulting in a low F1 score (0.3735).
Random Forest (Ensemble)	
Achieves the highest Accuracy (0.8435) and highest Precision (0.5217). However, its Recall (0.2553) is relatively low, meaning it misses many actual attrition cases. Its AUC (0.7893) is also strong, making it a good overall classifier.


## Run Streamlit link
https://project-folder-pgtjny3kfxappbfgfmmxhuy.streamlit.app

##Notebook name - 2025ac05932_ML_assignment.ipynb
