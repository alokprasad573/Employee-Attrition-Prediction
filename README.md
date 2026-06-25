# Employee Attrition Prediction

## Project Overview

Employee attrition is a critical challenge for organizations, leading to increased recruitment costs, productivity loss, and disruption of business operations. This project aims to identify employees at risk of leaving the organization and provide actionable insights to support employee retention strategies.

Using machine learning techniques, the project analyzes employee demographics, job-related factors, compensation, and workplace characteristics to predict attrition and assist HR teams in proactive decision-making.

---

## Business Problem

Employee turnover can negatively impact organizational performance and increase hiring costs. The objectives of this project are to:

- Identify key factors influencing employee attrition.
- Analyze employee behavior and attrition trends.
- Build predictive models to identify employees at risk of leaving.
- Provide data-driven recommendations to improve employee retention.

---

## Dataset Information

- Total Records: **1,470**
- Features Used: **18**
- Target Variable: **Attrition**
  - **0** → Employee Stays
  - **1** → Employee Leaves

### Class Distribution

| Class | Count | Percentage |
|---------|---------:|---------:|
| No Attrition | 1,233 | 83.9% |
| Attrition | 237 | 16.1% |

The dataset exhibits moderate class imbalance, requiring appropriate weighting techniques during model training.

---

## Exploratory Data Analysis (EDA)

### Key Findings

#### Overtime Strongly Influences Attrition

Employees working overtime were significantly more likely to leave the organization compared to employees who did not work overtime.

#### Salary Impacts Retention

Employees with lower monthly income showed higher attrition rates, indicating compensation is a major retention factor.

#### Younger Employees Leave More Frequently

Employees in lower age groups demonstrated higher attrition rates than senior employees.

#### Job Role Matters

The highest attrition rates were observed among:

- Sales Representatives
- Laboratory Technicians
- Human Resources Employees

These roles should receive priority attention in retention initiatives.

#### Job Satisfaction Affects Attrition

Employees reporting lower job satisfaction were more likely to leave the organization.

---

## Data Preprocessing

Several preprocessing steps were performed to prepare the data for machine learning models.

### Missing Value Analysis

- The dataset was checked for missing values.
- No missing values were found; therefore, imputation was not required.

### Outlier Analysis and Handling

Boxplots and descriptive statistical analysis were performed on numerical features to identify potential outliers.

Features such as:

- MonthlyIncome
- TotalWorkingYears
- YearsAtCompany
- DistanceFromHome

contained extreme observations.

These values were investigated and determined to represent valid employee records rather than data-entry errors. For example, employees with very high salaries or long work experience naturally appear as extreme observations.

Therefore:

- Outliers were **retained** to preserve important business information.
- No records were removed solely based on outlier values.
- Feature scaling was applied to minimize the influence of large numerical ranges.

### Feature Encoding

Categorical variables were transformed into numerical representations using one-hot encoding.

Examples:

| Original Feature | Encoded Feature |
|------------------|-----------------|
| MaritalStatus | MaritalStatus_Single |
| OverTime | OverTime_Yes |

### Feature Scaling

StandardScaler was applied to numerical features before training the Logistic Regression model.

Scaling standardized all numerical features to:

- Mean = 0
- Standard Deviation = 1

This ensured that features with larger magnitudes, such as MonthlyIncome, did not dominate model training.

### Train-Test Split

The dataset was split into:

- Training Set: 80%
- Testing Set: 20%

using stratified sampling to preserve the original class distribution.

---

## Feature Selection & Class Imbalance Handling

### Feature Selection

Feature importance analysis and exploratory data analysis were used to identify the variables contributing most significantly to employee attrition.

The most influential features included:

| Feature | Business Impact |
|----------|----------|
| OverTime | Employees working overtime were more likely to leave. |
| MonthlyIncome | Lower income was associated with higher attrition. |
| Age | Younger employees showed higher turnover rates. |
| JobSatisfaction | Lower satisfaction increased attrition risk. |
| YearsAtCompany | Employees with shorter tenure were more likely to leave. |
| TotalWorkingYears | Less experienced employees showed higher attrition. |
| MaritalStatus | Single employees exhibited higher attrition rates. |
| JobLevel | Lower job levels were associated with increased attrition. |

These features provided the highest predictive value and were retained for model training.

---

### Handling Class Imbalance

The dataset was imbalanced:

| Class | Count | Percentage |
|---------|---------:|---------:|
| No Attrition (0) | 1,233 | 83.9% |
| Attrition (1) | 237 | 16.1% |

To improve prediction performance for attrition cases, class-weighting techniques were applied.

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

Attrition cases received approximately **5.2 times more importance** than non-attrition cases during model training.

#### XGBoost Scale Positive Weight

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

This helped XGBoost focus more effectively on the minority attrition class.

### Why Class Weighting Was Necessary

Without weighting, models would tend to favor the majority class because nearly 84% of employees remained with the company.

Class weighting improved the model's ability to identify employees at risk of leaving and significantly improved Recall.

---

## Machine Learning Models

The following machine learning models were trained and evaluated:

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

- Achieved the highest ROC-AUC score (**0.747**)
- Achieved the highest Recall (**76.6%**)
- Correctly identified **36 out of 47 attrition cases**
- Missed the fewest employees likely to leave
- Demonstrated strong generalization performance
- Showed minimal overfitting
- Provides interpretable results for HR stakeholders

---

## Business Insights

The strongest drivers of employee attrition were:

- Excessive overtime
- Lower monthly income
- Lower job satisfaction
- Younger age
- Shorter tenure
- High-risk job roles

These insights can help HR teams proactively identify and retain at-risk employees.

---

## Recommendations

### Reduce Overtime

Implement workload management and work-life balance initiatives for employees frequently working overtime.

### Review Compensation Policies

Conduct salary benchmarking and compensation reviews for employees in lower income brackets.

### Improve Career Development

Provide structured career growth opportunities and internal mobility programs.

### Focus on High-Risk Roles

Develop targeted retention strategies for:

- Sales Representatives
- Laboratory Technicians
- HR Employees

### Monitor Employee Satisfaction

Regular engagement surveys can help identify dissatisfaction before it results in employee turnover.

---

## Limitations

- The model generated false positives, which may lead to unnecessary retention efforts.
- Some attrition cases were not identified.
- External factors such as organizational culture, leadership quality, and job market conditions were not included.
- Employee behavior changes over time, requiring periodic model retraining.
- Predictions should support HR decisions rather than replace human judgment.

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

This project successfully developed an employee attrition prediction system capable of identifying employees at risk of leaving the organization.

The analysis revealed that overtime, compensation, job satisfaction, age, and job role are the strongest drivers of attrition. Among all evaluated models, **Logistic Regression** delivered the best balance between predictive performance, interpretability, and business value.

The resulting solution can serve as an early-warning system to help HR teams proactively improve employee retention and reduce turnover costs.
