# Employee Turnover Analysis

## Project Overview
This project aims to analyze and predict employee turnover within a large U.S. company. Utilizing a dataset of 9,540 employee records, we applied Logistic Regression, KNN Classifier, and Random Forest models to uncover patterns and factors contributing to employee churn.

## Goals
- To identify the main factors that contribute to employee turnover.
- To predict potential turnovers using machine learning models.
- To provide data-driven recommendations for retention strategies.

## Dataset
The dataset includes 9,540 observations with the following features:
- Numerical: Review scores, number of projects, tenure, satisfaction level, and average monthly hours.
- Categorical: Department, received bonus, promotion status, salary level, and turnover status.

## Technical Details
- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Models**: Logistic Regression, K-Nearest Neighbors, Random Forest
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC

## Technical Workflow
- **Encoding Categorical Variables**: Transformation of categorical data using label encoding and ordinal encoding.
- **Data Splitting**: Segmentation of data into training and testing sets for model validation.
- **Data Scaling**: Standardization of features using `StandardScaler` to normalize the data distribution.
- **Model Optimization**: Utilization of `GridSearchCV` to find the optimal 'k' value and hyperparameters for the KNN model.
- **Model Evaluation**: Performance assessment through in-sample and out-of-sample accuracy, precision, and recall metrics.

## Key Findings
- The optimal KNN configuration was achieved with `k=14`, reaching an accuracy rate of 84.4% with uniform weights, slightly improved to 84.5% with distance weights.
- Through detailed analysis, including confusion matrix and feature importance evaluation, the model demonstrates strong predictive capabilities without overfitting.
- Visualization of decision boundaries pre and post-application of SMOTE (Synthetic Minority Over-sampling Technique) for balanced dataset representation.
