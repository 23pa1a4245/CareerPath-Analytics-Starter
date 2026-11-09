# CareerPath Analytics — Viva Preparation

## 1. What is the objective?
To analyze career-related employee indicators and workforce patterns, then present insights that can inform career guidance and workforce planning.

## 2. Why did you choose this dataset?
It contains job level, work experience, years in the current role, time since the last promotion, training frequency, salary, department, and attrition. It is a useful educational dataset, though it is synthetic.

## 3. Why use Streamlit?
Streamlit lets us build an interactive Python dashboard quickly, with filters, charts, metrics, and data downloads.

## 4. What is exploratory data analysis?
EDA is the process of inspecting data, checking distributions and missing values, calculating summaries, and visualizing patterns before drawing conclusions.

## 5. What is the machine-learning target?
The target is `Attrition`: Yes means the record is labelled as an employee who left; No means the employee stayed. The model does not predict promotions.

## 6. Why compare Logistic Regression and Random Forest?
Logistic Regression is a simple baseline classifier. Random Forest can capture nonlinear relationships and interactions. Comparing them provides a more informed model choice.

## 7. What is class imbalance?
It means one target class is much more common than another. In this dataset, employees who stayed are more common than employees who left, so accuracy alone may be misleading.

## 8. Explain precision and recall.
Precision asks: among records predicted as leaving, what proportion were actually labelled as leaving? Recall asks: among records labelled as leaving, what proportion did the model identify?

## 9. What is ROC-AUC?
ROC-AUC summarizes how well a classifier ranks positive cases above negative cases across different thresholds. It should be interpreted with other metrics and the dataset's limitations.

## 10. Why use a train-test split?
To evaluate model performance on records not used to fit the model, providing a better estimate of generalization than measuring only training performance.

## 11. Can your project predict promotions?
No. The dataset has promotion-related indicators but lacks verified promotion outcomes and chronological promotion histories. The current ML task is attrition classification.

## 12. What are the limitations?
The data is synthetic and static, the model may not generalize to real organizations, the features may contain bias, and correlations do not prove causation. The model must not be used for real employment decisions.

## 13. How does the project support workforce planning?
It summarizes department and job-level distribution, training, compensation, and attrition patterns to identify areas that deserve further review.

## 14. What future improvements would you make?
Use governed longitudinal employee data, actual promotion history, validated skills data, fairness and privacy reviews, model calibration, external validation, and carefully evaluated workforce forecasting.
