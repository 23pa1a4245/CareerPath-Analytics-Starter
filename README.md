<<<<<<< HEAD
# CareerPath Analytics

**Employee Career Progression, Career Guidance & Workforce Planning**

An educational data analytics and machine-learning project built with Python, Pandas, scikit-learn, Plotly, and Streamlit.

## Features
- Executive dashboard with workforce KPIs and department/job-level filters
- Career progression indicators: years in current role and years since last promotion
- Workforce planning analysis: job levels, departments, training, and compensation
- Machine-learning comparison: Logistic Regression and Random Forest for attrition classification
- Model evaluation: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
- Group-level career-development review indicators
- CSV download for filtered data
- Responsible-use notes and limitations

## Dataset
Use the IBM HR Analytics Employee Attrition & Performance dataset:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Download and extract the CSV file, then rename it to `hr_data.csv` and place it in the project root beside `app.py`.

The dataset is fictional/synthetic and is used here for educational exploration.

## Setup in VS Code

```bash
python -m pip install -r requirements.txt
```

Train and evaluate the machine-learning models:

```bash
python train_model.py
```

Start the dashboard:

```bash
streamlit run app.py
```

On Windows, if `python` is not recognized, try `py -m pip install -r requirements.txt`, `py train_model.py`, and `py -m streamlit run app.py`.

## Project structure

```text
CareerPath-Analytics/
├── app.py
├── train_model.py
├── hr_data.csv                 # Add the downloaded dataset here
├── requirements.txt
├── README.md
├── PROJECT_REPORT_TEMPLATE.md
├── VIVA_QUESTIONS.md
├── models/                     # Model artifacts are generated here
└── screenshots/                # Save dashboard screenshots here
```

## Methodology
1. Load and inspect employee data.
2. Clean target values and select relevant features.
3. Exclude identifiers and selected demographic fields from the demonstration model.
4. Split data into training and held-out test sets using stratification.
5. Train Logistic Regression and Random Forest pipelines with imputation, encoding, and scaling.
6. Compare models using several classification metrics and select by test ROC-AUC.
7. Explore aggregate career and workforce patterns in the Streamlit dashboard.

## Important limitations
- The dataset is a static snapshot; it does not contain chronological promotion events.
- The model predicts the dataset's `Attrition` label, not promotion eligibility or career success.
- The data is synthetic and does not establish real-world predictive validity.
- Do not use this classroom project for individual employment decisions.
- Observed associations do not establish causation.
=======
# CareerPath-Analytics-Starter
>>>>>>> 6c66d1bd01d474c8aaa8b6732d287e9687dbf684
