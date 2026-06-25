# Employee Attrition Prediction

## Project Overview

Employee attrition is one of the most significant challenges faced by organizations, leading to increased recruitment costs, productivity loss, and knowledge gaps. This project aims to identify employees at risk of leaving the organization and provide actionable insights to improve employee retention.

Using machine learning techniques, the project analyzes employee demographics, job-related factors, compensation, and work-life characteristics to predict attrition and support HR decision-making.

---

## Business Problem

Employee turnover can negatively impact organizational performance. The objective of this project is to:

- Identify factors contributing to employee attrition.
- Analyze employee behavior and attrition trends.
- Build predictive models to identify employees at risk of leaving.
- Provide HR teams with data-driven retention recommendations.

---

## Dataset Information

- Total Records: **1,470**
- Features: **18**
- Target Variable: **Attrition**
  - **0** → Employee Stays
  - **1** → Employee Leaves

### Class Distribution

| Class | Count | Percentage |
|---------|---------:|---------:|
| No Attrition | 1,233 | 83.9% |
| Attrition | 237 | 16.1% |

The dataset is moderately imbalanced, requiring class-weighting techniques during model training.

---

## Exploratory Data Analysis (EDA)

### Key Findings

#### 1. Overtime Strongly Influences Attrition

Employees working overtime were significantly more likely to leave the organization compared to employees who did not work overtime.

#### 2. Salary Impacts Retention

Employees with lower monthly income showed higher attrition rates, indicating compensation plays a major role in retention.

#### 3. Younger Employees Leave More Frequently

Employees in lower age groups demonstrated higher attrition rates than senior employees.

#### 4. Job Role Matters

The highest attrition rates were observed among:

- Sales Representatives
- Laboratory Technicians
- Human Resources Employees

These roles should receive priority attention in retention strategies.

#### 5. Job Satisfaction Affects Attrition

Employees reporting lower job satisfaction were more likely to leave the organization.

---

## Data Preprocessing

The following preprocessing steps were performed:

- Handling categorical variables using encoding techniques.
- Feature scaling where required.
- Train-Test Split (80:20).
- Class imbalance handling using class weighting.
- Feature selection and preparation for machine learning models.

---

## Feature Selection & Class Imbalance Handling

### Feature Selection

Feature importance analysis and exploratory data analysis were used to identify the variables that contributed most to employee attrition.

The most influential features included:

| Feature | Business Impact |
|----------|----------|
| OverTime | Employees working overtime were more likely to leave. |
| MonthlyIncome | Lower income was associated with higher attrition. |
| Age | Younger employees showed higher turnover rates. |
| JobSatisfaction | Lower satisfaction increased attrition risk. |
| YearsAtCompany | Employees with shorter tenure were more likely to leave. |
| JobRole | Certain roles exhibited significantly higher attrition. |
| TotalWorkingYears | Less experienced employees showed higher attrition. |
| WorkLifeBalance | Poor work-life balance increased attrition risk. |

These features contributed the most predictive power and were retained during model training.

---

### Handling Class Imbalance

The dataset was imbalanced:

| Class | Count | Percentage |
|---------|---------:|---------:|
| No Attrition (0) | 1,233 | 83.9% |
| Attrition (1) | 237 | 16.1% |

To ensure the models paid sufficient attention to employees who left the company, class weighting techniques were applied.

#### Logistic Regression Class Weights

Weights were calculated using:

```python
weight = Total Samples / (Number of Classes × Class Count)
```

Training Data:

| Class | Count | Weight |
|---------|---------:|---------:|
| No Attrition (0) | 986 | 0.596 |
| Attrition (1) | 190 | 3.095 |

This means attrition cases received approximately **5.2 times more importance** than non-attrition cases during training.

#### XGBoost Scale Positive Weight

For XGBoost:

```python
scale_pos_weight = Negative Samples / Positive Samples
```

Calculation:

```python
986 / 190 = 5.19
```

Therefore:

```python
scale_pos_weight = 5.19
```

This adjustment helped the model focus more effectively on the minority attrition class.

---

### Why Class Weighting Was Necessary

Without class weighting, the models would tend to predict the majority class (No Attrition) because approximately 84% of employees stayed with the company.

Applying class weights improved the model's ability to identify employees at risk of leaving and increased Recall, which is a critical metric for attrition prediction.

## Machine Learning Models

The following models were trained and evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

---

## Model Performance Comparison

| Metric | Logistic Regression | Random Forest | XGBoost |
|----------|----------:|----------:|----------:|
| Accuracy | 73.47% | 80.61% | 80.61% |
| Precision | 35.0% | **42.2%** | 41.9% |
| Recall | **76.6%** | 57.4% | 55.3% |
| F1 Score | 48.0% | **48.6%** | 47.7% |
| ROC-AUC | **0.747** | 0.712 | 0.704 |

---

## Confusion Matrix Summary

| Metric | Logistic Regression | Random Forest | XGBoost |
|----------|----------:|----------:|----------:|
| True Negatives (TN) | 180 | 210 | 211 |
| False Positives (FP) | 67 | 37 | 36 |
| False Negatives (FN) | **11** | 20 | 21 |
| True Positives (TP) | **36** | 27 | 26 |

---

## Selected Model: Logistic Regression

Although Random Forest and XGBoost achieved higher accuracy, Logistic Regression was selected as the final model because it:

- Achieved the highest ROC-AUC score (**0.747**).
- Achieved the highest Recall (**76.6%**).
- Correctly identified **36 out of 47 employees** who left.
- Missed the fewest attrition cases.
- Demonstrated minimal overfitting.
- Provides strong interpretability for HR stakeholders.

---

## Business Insights

The analysis identified the following major attrition drivers:

- Excessive overtime
- Lower monthly income
- Lower job satisfaction
- Younger employees
- High-risk job roles

These factors can be used to proactively identify employees who may be at risk of leaving.

---

## Recommendations

### 1. Reduce Overtime

Implement workload monitoring and work-life balance initiatives for employees regularly working overtime.

### 2. Review Compensation Policies

Conduct salary benchmarking and compensation reviews for employees in lower pay brackets.

### 3. Improve Career Growth Opportunities

Provide structured career progression plans and professional development programs.

### 4. Focus on High-Risk Roles

Develop targeted retention strategies for:

- Sales Representatives
- Laboratory Technicians
- HR Staff

### 5. Monitor Employee Satisfaction

Regular employee engagement surveys can help identify dissatisfaction before it leads to attrition.

---

## Limitations

- The model generated false positives, which may lead to unnecessary retention efforts.
- Some attrition cases were not identified.
- The dataset does not include external factors such as market opportunities or organizational culture.
- Employee behavior changes over time, requiring periodic model retraining.
- Predictions should be used as decision-support tools rather than automated decision systems.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Jupyter Notebook

---

## Conclusion

This project successfully developed an employee attrition prediction system capable of identifying employees at risk of leaving the organization. The analysis revealed that overtime, salary, job satisfaction, and specific job roles are the strongest drivers of attrition.

Among all evaluated models, **Logistic Regression** delivered the best balance between predictive performance, interpretability, and business value, making it the most suitable solution for supporting HR retention initiatives.
