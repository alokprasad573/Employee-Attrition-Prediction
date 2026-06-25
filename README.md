# Employee Attrition Prediction

> A machine learning project to identify employees at risk of leaving — enabling HR teams to take proactive retention actions before it's too late.

---

## Table of Contents

- [Overview](#overview)
- [Business Objective](#business-objective)
- [Project Structure](#project-structure)
- [Dataset Information](#dataset-information)
- [Exploratory Data Analysis](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Feature Selection](#feature-selection)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Machine Learning Models](#machine-learning-models)
- [Model Performance Comparison](#model-performance-comparison)
- [Confusion Matrix Summary](#confusion-matrix-summary)
- [Final Model Selection](#final-model-selection)
- [Business Insights & Recommendations](#business-insights--recommendations)
- [Limitations](#limitations)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Conclusion](#conclusion)

---

## Overview

Employee attrition is a major challenge for organizations, resulting in increased recruitment costs, productivity loss, and disruption of business operations. This project uses machine learning to identify employees at risk of leaving and provides actionable insights to support HR retention strategies.

The project combines **Exploratory Data Analysis (EDA)**, **feature engineering**, **machine learning model development**, and **business-focused recommendations** to help organizations proactively reduce employee turnover.

---

## Business Objective

The primary objectives of this project are to:

- Identify key factors influencing employee attrition.
- Analyze employee behavior and attrition trends.
- Build predictive models to identify employees at risk of leaving.
- Provide HR teams with actionable recommendations to improve retention.

---

## Project Structure

```
Employee-Attrition-Prediction/
│
├── data/
│   ├── employee_attrition.csv           # Raw dataset (1,470 records)
│   ├── cleaned_employee_attrition.csv   # Preprocessed & cleaned dataset
│   └── training_dataset.csv             # Final dataset used for model training
│
├── models/
│   ├── logistic_regression_model.pkl    # Trained Logistic Regression model
│   ├── random_forest_model.pkl          # Trained Random Forest model
│   ├── xgb_model.pkl                    # Trained XGBoost model
│   └── scaler.pkl                       # Fitted StandardScaler for inference
│
├── notebooks/
│   ├── EDA.ipynb                        # Exploratory Data Analysis notebook
│   ├── TRAINING.ipynb                   # Model training & evaluation notebook
│   ├── REPORT.ipynb                     # Final report notebook
│   └── plots/                           # Saved visualization outputs
│       ├── LR_HeatMap.png
│       ├── attrition vs Department.png
│       ├── attrition vs job role.png
│       ├── attrition vs monthlyincome.png
│       ├── comparisions.png
│       └── top_features.png
│
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependencies
├── Employee Attrition Analysis.pdf      # Project analysis report
└── README.md
```

---

## Dataset Information

| Attribute | Value |
|-----------|-------|
| Total Records | 1,470 |
| Features Used | 18 |
| Target Variable | Attrition |
| Problem Type | Binary Classification |

### Target Distribution

| Class | Count | Percentage |
|-------|------:|-----------:|
| No Attrition (0) | 1,233 | 83.9% |
| Attrition (1) | 237 | 16.1% |

The dataset is **moderately imbalanced**, requiring class-weighting techniques during model training.

---

## Exploratory Data Analysis (EDA)

### Key Insights

#### Overtime Increases Attrition Risk
Employees working overtime were significantly more likely to leave the organization.

#### Compensation Influences Retention
Lower monthly income was strongly associated with higher attrition rates.

#### Younger Employees Are More Likely to Leave
Employees in lower age groups showed higher turnover rates compared to experienced employees.

#### Job Role Matters
The highest attrition rates were observed among:
- Sales Representatives
- Laboratory Technicians
- Human Resources Employees

#### Job Satisfaction Impacts Attrition
Employees reporting lower job satisfaction exhibited a higher likelihood of leaving.

---

## Data Preprocessing

### Missing Value Analysis
- Dataset inspected for missing values.
- No missing values were identified.

### Outlier Analysis & Handling

Numerical features were analyzed using boxplots and descriptive statistics. Potential outliers were identified in:

- MonthlyIncome
- TotalWorkingYears
- YearsAtCompany
- DistanceFromHome

These observations represented **valid employee records** rather than data-entry errors. Therefore:

- No records were removed.
- Outliers were retained to preserve business information.
- `StandardScaler` was applied to reduce the impact of varying feature magnitudes.

### Feature Encoding

Categorical variables were converted into numerical format using one-hot encoding.

| Original Feature | Encoded Feature |
|------------------|-----------------|
| MaritalStatus | MaritalStatus_Single |
| OverTime | OverTime_Yes |

### Feature Scaling

`StandardScaler` was applied to numerical variables before training Logistic Regression.

**Benefits:**
- Improved model convergence
- Reduced influence of feature magnitude differences
- Enhanced model stability

### Train-Test Split

| Split | Ratio |
|-------|-------|
| Training Set | 80% |
| Testing Set | 20% |

Stratified sampling was used to preserve class distribution across both splits.

---

## Feature Selection

Feature importance analysis and business understanding were used to identify the most influential variables.

### Selected Features (18 Total)

| # | Feature |
|---|---------|
| 1 | Age |
| 2 | DistanceFromHome |
| 3 | EnvironmentSatisfaction |
| 4 | JobInvolvement |
| 5 | JobLevel |
| 6 | JobSatisfaction |
| 7 | MonthlyIncome |
| 8 | NumCompaniesWorked |
| 9 | PercentSalaryHike |
| 10 | RelationshipSatisfaction |
| 11 | StockOptionLevel |
| 12 | TotalWorkingYears |
| 13 | YearsAtCompany |
| 14 | YearsInCurrentRole |
| 15 | YearsSinceLastPromotion |
| 16 | YearsWithCurrManager |
| 17 | MaritalStatus_Single |
| 18 | OverTime_Yes |

### Most Influential Drivers of Attrition

| Feature | Impact |
|---------|--------|
| OverTime | Higher overtime increases attrition risk |
| MonthlyIncome | Lower income increases attrition risk |
| Age | Younger employees are more likely to leave |
| JobSatisfaction | Lower satisfaction increases attrition |
| YearsAtCompany | Lower tenure increases attrition |
| TotalWorkingYears | Less experienced employees leave more frequently |

---

## Handling Class Imbalance

### Logistic Regression — Class Weight Calculation

Training set distribution:

| Class | Count |
|-------|------:|
| No Attrition (0) | 986 |
| Attrition (1) | 190 |

Class weights were calculated using Scikit-Learn's balanced weighting formula:

```python
weight = Total Samples / (Number of Classes × Samples in Class)
```

Where:

```python
Total Samples    = 1176
Number of Classes = 2
```

For employees who stayed (Class 0):

```python
weight_0 = 1176 / (2 × 986)
weight_0 = 0.596
```

For employees who left (Class 1):

```python
weight_1 = 1176 / (2 × 190)
weight_1 = 3.095
```

Final class weights:

```python
class_weight = {
    0: 0.596,
    1: 3.095
}
```

Implementation:

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    penalty='l2',
    solver='lbfgs',
    class_weight={
        0: 0.596,
        1: 3.095
    },
    max_iter=1000,
    random_state=42
)
```

#### Why This Was Important

Without weighting, the model would naturally favor the majority class because approximately **84% of employees stayed** with the company.

By assigning a higher weight to attrition cases:

- Misclassifying an employee who leaves becomes **more costly**.
- The model pays greater attention to **minority-class observations**.
- **Recall improves significantly.**
- More at-risk employees are identified for potential retention actions.

As a result, Logistic Regression successfully identified **36 out of 47 employees who left**, achieving a Recall of **76.6%** — the highest among all evaluated models.

Attrition cases received approximately **5.2× higher importance** during training.

---

### XGBoost — Scale Positive Weight

```python
scale_pos_weight = 986 / 190
scale_pos_weight = 5.19
```

This ensured that the minority attrition class received greater emphasis during XGBoost model training.

---

## Machine Learning Models

The following models were trained and evaluated:

| # | Model |
|---|-------|
| 1 | Logistic Regression |
| 2 | Random Forest Classifier |
| 3 | XGBoost Classifier |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | Of predicted leavers, how many actually left |
| Recall | Of actual leavers, how many were correctly identified |
| F1 Score | Harmonic mean of Precision and Recall |
| ROC-AUC | Overall discriminative ability of the model |

> **Recall** was prioritized as the primary metric, since missing an employee who actually leaves (False Negative) has greater business cost than a false alarm.

---

## Model Performance Comparison

### Test Set Metrics (Unseen Data)

| Metric | Logistic Regression | Random Forest | XGBoost |
|--------|--------------------:|--------------:|--------:|
| Accuracy | 0.7347 | 0.8061 | 0.8061 |
| Precision | 0.350 | **0.422** | 0.419 |
| Recall | **0.766** | 0.574 | 0.553 |
| F1 Score | 0.480 | **0.486** | 0.477 |
| ROC-AUC | **0.747** | 0.712 | 0.704 |

### Training Set Metrics

| Metric | Logistic Regression | Random Forest | XGBoost |
|--------|--------------------:|--------------:|--------:|
| Accuracy | 0.7483 | 0.8640 | **0.9345** |
| Precision | 0.3668 | 0.5630 | **0.7233** |
| Recall | **0.7684** | 0.7053 | 0.9632 |
| F1 Score | 0.4966 | 0.6262 | **0.8262** |
| ROC-AUC | 0.7564 | 0.7999 | **0.9461** |

> The large gap between XGBoost's training metrics (0.93+ accuracy) and test metrics (0.81) confirms significant **overfitting**. Logistic Regression shows consistent performance across both splits, indicating strong generalization.

---

### Logistic Regression

| Split | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|--------:|---------:|-------:|--------:|--------:|
| Training | 0.7483 | 0.3668 | 0.7684 | 0.4966 | 0.756 |
| Testing | 0.7347 | 0.3500 | 0.7660 | 0.4800 | 0.747 |
| **Gap** | **0.0136** | **0.0168** | **0.0024** | **0.0166** | **0.009** |

**Strengths**
- Highest ROC-AUC score on test data
- Highest Recall — catches most at-risk employees
- Minimal gap between train and test scores (no overfitting)
- Highly interpretable for HR stakeholders

---

### Random Forest

| Split | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|--------:|---------:|-------:|--------:|--------:|
| Training | 0.8640 | 0.5630 | 0.7053 | 0.6262 | 0.800 |
| Testing | 0.8061 | 0.4219 | 0.5745 | 0.4865 | 0.712 |
| **Gap** | **0.0579** | **0.1411** | **0.1308** | **0.1397** | **0.088** |

**Strengths**
- Highest Precision on test data
- Good overall accuracy

**Limitations**
- Missed more attrition cases than Logistic Regression
- Moderate overfitting observed (train-test gap in recall: ~0.13)

---

### XGBoost

| Split | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|--------:|---------:|-------:|--------:|--------:|
| Training | 0.9345 | 0.7233 | 0.9632 | 0.8262 | 0.946 |
| Testing | 0.8061 | 0.4194 | 0.5532 | 0.4771 | 0.704 |
| **Gap** | **0.1284** | **0.3039** | **0.4100** | **0.3491** | **0.242** |

**Strengths**
- Competitive performance
- Strong gradient boosting capability

**Limitations**
- Significant overfitting observed (train recall 0.963 vs test recall 0.553)
- Did not outperform Logistic Regression on unseen data

---

## Confusion Matrix Summary

| Metric | Logistic Regression | Random Forest | XGBoost |
|--------|--------------------:|--------------:|--------:|
| True Negatives (TN) | 180 | 210 | 211 |
| False Positives (FP) | 67 | 37 | 36 |
| False Negatives (FN) | **11** | 20 | 21 |
| True Positives (TP) | **36** | 27 | 26 |

Logistic Regression produced the **fewest False Negatives (11)**, meaning it missed the fewest actual attrition cases — the most critical error type in this business context.

---

## Final Model Selection

### ✅ Selected: Logistic Regression

Although Random Forest and XGBoost achieved higher accuracy, **Logistic Regression** was selected as the final model because:

| Criterion | Value |
|-----------|-------|
| ROC-AUC | **0.747** (highest) |
| Recall | **76.6%** (highest) |
| False Negatives | **11** (lowest) |
| Interpretability | High — suitable for HR reporting |
| Overfitting | Minimal |

> In an HR context, identifying as many at-risk employees as possible (high Recall) is more valuable than high Accuracy, which can be misleadingly inflated by the majority class.

---

## Business Insights & Recommendations

### Key Drivers of Attrition

| Driver | Observation |
|--------|-------------|
| Overtime | Employees working overtime are significantly more likely to leave |
| Monthly Income | Lower-paid employees leave more frequently |
| Age | Younger employees show higher turnover |
| Job Satisfaction | Low satisfaction strongly predicts attrition |
| Tenure | Newer employees are at higher risk |
| Job Role | Sales Reps, Lab Technicians, HR staff are highest-risk |

### Recommendations

#### Reduce Excessive Overtime
Monitor workload distribution and promote work-life balance initiatives to reduce burnout.

#### Improve Compensation Strategies
Conduct salary benchmarking and periodic compensation reviews to stay competitive.

#### Strengthen Career Development
Provide structured career progression paths and internal mobility programs for early-career employees.

#### Focus on High-Risk Roles
Develop targeted retention programs specifically for:
- Sales Representatives
- Laboratory Technicians
- Human Resources Employees

#### Improve Employee Engagement
Deploy regular satisfaction surveys and feedback programs to detect disengagement early — before resignation occurs.

---

## Limitations

- **False Positives** may result in unnecessary retention efforts being directed at employees not planning to leave.
- **Some attrition cases remain unidentified** (False Negatives) due to the inherent trade-off between Recall and Precision.
- **External factors** such as organizational culture, job market conditions, and macroeconomic variables were not available in the dataset.
- **Model drift** may occur as employee behavior and company conditions change over time — requiring periodic retraining.
- **Predictions should support HR decisions**, not replace human judgment.

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-000000?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

### Versions

| Library | Version |
|---------|---------|
| streamlit | 1.58.0 |
| pandas | 3.0.3 |
| numpy | 2.5.0 |
| scikit-learn | 1.9.0 |
| xgboost | 3.3.0 |
| joblib | 1.5.3 |
| plotly | 5.18.0 |
| matplotlib | 3.11.0 |
| seaborn | 0.13.2 |

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

### 5. Explore the Notebooks

Open the notebooks in the `notebooks/` directory using Jupyter:

```bash
jupyter notebook
```

| Notebook | Purpose |
|----------|---------|
| `EDA.ipynb` | Exploratory Data Analysis |
| `TRAINING.ipynb` | Model training & evaluation |
| `REPORT.ipynb` | Final analysis report |

---

## Conclusion

This project successfully developed an **employee attrition prediction system** capable of identifying employees at risk of leaving the organization.

The analysis revealed that **overtime, compensation, age, job satisfaction, and tenure** are the strongest drivers of attrition. Among all evaluated models, **Logistic Regression** provided the best balance of predictive performance, generalization capability, and interpretability — identifying **76.6% of employees who left** while remaining easy to explain to non-technical HR stakeholders.

The resulting solution can serve as an **early-warning system** to help HR teams proactively improve retention and reduce employee turnover costs.