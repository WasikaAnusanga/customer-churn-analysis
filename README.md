# Customer Churn Prediction and Retention Analysis

## DS 01 — Advanced Data Science Assignment

An end-to-end machine learning project for predicting customer churn, identifying high-risk customers, and supporting data-driven customer retention strategies.

The project includes data auditing, exploratory data analysis, feature engineering, machine learning, hyperparameter optimization, probability calibration, explainability, robustness testing, and business impact simulation.

---

## 1. Project Overview

Customer churn is an important challenge for subscription-based businesses. Identifying customers who are likely to leave enables companies to prioritize retention activities and allocate resources more efficiently.

This project develops a machine learning solution that estimates customer churn probabilities and ranks customers according to their predicted risk.

The business scenario assumes that the company can contact only the highest-risk 20% of customers during each retention cycle.

Rather than focusing exclusively on prediction accuracy, this project emphasizes:

- Reliable churn-risk ranking
- Model discrimination and calibration
- Explainable machine learning
- Data leakage prevention
- Business-oriented evaluation
- Customer retention cost-benefit analysis

**Final Research Model:** Sigmoid-Calibrated Random Forest

## 2. Project Objectives

The main objectives are to:

1. Audit and clean the customer churn dataset.
2. Investigate customer churn patterns using exploratory data analysis.
3. Develop meaningful engineered features.
4. Build reproducible preprocessing and machine learning pipelines.
5. Compare Logistic Regression, Random Forest, and Gradient Boosting.
6. Optimize hyperparameters using stratified cross-validation.
7. Investigate class imbalance and classification thresholds.
8. Evaluate probability calibration and model reliability.
9. Explain global feature importance and individual customer predictions.
10. Simulate retention targeting under a 20% contact-capacity constraint.
11. Estimate hypothetical retention benefits.
12. Evaluate robustness and independent test performance.

## 3. Dataset Information

The project uses `customer_churn_advanced_dataset.csv`.

| Property | Value |
|---|---|
| Total Customers | 1,200 |
| Total Columns | 20 |
| Target Variable | churn |
| Churned Customers | 1,036 (86.33%) |
| Retained Customers | 164 (13.67%) |
| Training Customers | 960 |
| Testing Customers | 240 |
| Missing Cells | 138 |
| Duplicate Customer IDs | 0 |

The target variable is encoded as:

- `1` — Customer churned
- `0` — Customer retained

The dataset contains customer demographic information, contract types, billing details, service usage, satisfaction scores, complaints, and payment behavior.

### Data Quality Findings

The original dataset contains 138 missing cells across four numerical variables.

Additionally:

- 147 customers have positive monthly usage despite having internet service recorded as "No".
- Five records exhibit large differences between total charges and simplified billing estimates.
- No duplicate customer identifiers were identified.

These anomalies were documented rather than automatically deleted.

Missing values remain in the exported cleaned dataset because statistical imputation is performed safely within the machine learning pipeline.

## 4. Project Structure

The project is organized as follows:

```text
DS01_Customer_Churn/
|
|-- data/
|   |-- raw/
|   |   |-- customer_churn_advanced_dataset.csv
|   |
|   |-- processed/
|       |-- 02_cleaned_dataset.csv
|
|-- notebooks/
|   |-- 01_data_science_notebook.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- feature_engineering.py
|   |-- preprocessing.py
|   |-- modeling.py
|   |-- robustness.py
|
|-- models/
|   |-- 03_model_artifact.joblib
|   |-- model_metadata.json
|
|-- figures/
|   |-- 01_churn_distribution.png
|   |-- ...
|   |-- 21_final_test_confusion_matrix.png
|
|-- reports/
|   |-- 04_business_report.docx
|   |-- 04_business_report.pdf
|   |-- final_test_metrics.csv
|   |-- final_test_predictions.csv
|   |-- data_quality_flags.csv
|   |-- model_robustness_comparison.csv
|   |-- Other analysis result files
|
|-- 05_README.md
|-- requirements.txt
|-- .gitignore
```

### Important Files

**01_data_science_notebook.ipynb**

Contains the complete executable machine learning workflow, including visualizations, model development, evaluation, explainability, and retention analysis.

**02_cleaned_dataset.csv**

Contains the standardized dataset with all 1,200 customer records. Missing numerical values are preserved for pipeline-based imputation.

**03_model_artifact.joblib**

Contains the trained calibrated Random Forest pipeline, including feature engineering, preprocessing, classification, and probability calibration.

**04_business_report.pdf**

Contains the professional eight-page report presenting methodology, results, business insights, limitations, and recommendations.

**05_README.md**

Contains project documentation and execution instructions.

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and auditing |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and evaluation |
| Joblib | Model serialization |
| Jupyter Notebook | Executable analysis |
| Visual Studio Code | Development environment |

Exact installed package versions are documented in `requirements.txt`.

## 6. Installation and Setup

### Prerequisites

- Windows computer
- Python 3 installed
- Visual Studio Code
- Python and Jupyter VS Code extensions

### Step 1 — Open the Project

Open the `DS01_Customer_Churn` folder in Visual Studio Code.

### Step 2 — Create a Virtual Environment

Open the VS Code terminal and run:

```powershell
py -m venv .venv
```

### Step 3 — Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4 — Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 5 — Open the Notebook

Open:

```text
notebooks/01_data_science_notebook.ipynb
```

Select the Python interpreter associated with the `.venv` environment.

### Step 6 — Execute the Notebook

Use **Run All** in Jupyter Notebook to execute the complete analysis sequentially.

Run the notebook with the current working directory set to the project root or the `notebooks` folder so the dataset paths resolve correctly.

The full execution may take several minutes because hyperparameter searches and calibrated cross-validation train multiple machine learning models.

## 7. Machine Learning Methodology

### 7.1 Data Preparation

The dataset was divided using a stratified 80:20 train-test split.

- Training dataset: 960 customers
- Testing dataset: 240 customers
- Random seed: 42

The final test dataset was held out from exploratory model development and hyperparameter optimization.

### 7.2 Feature Engineering

Eight engineered features were created:

1. tenure_years
2. payment_risk_flag
3. complaint_intensity
4. total_service_issues
5. monthly_charge_per_gb
6. month_to_month_late_payments
7. low_satisfaction_flag
8. charge_satisfaction_interaction

These features were designed to represent customer tenure, payment risk, complaint behavior, billing, satisfaction, and interactions.

The target variable was not used to construct engineered features.

### 7.3 Preprocessing

A reproducible Scikit-learn preprocessing pipeline was implemented.

Numerical preprocessing includes:

- Median missing-value imputation
- Missing-value indicators
- Standard scaling

Categorical preprocessing includes:

- Most-frequent imputation
- One-hot encoding

All learned preprocessing transformations were performed within training folds during cross-validation.

Customer identifiers and the potentially leakage-prone `total_charges` field were excluded from predictive modeling.

### 7.4 Classification Models

Three machine learning classifiers were developed:

- Logistic Regression
- Random Forest
- Gradient Boosting

A majority-class DummyClassifier was also included as a benchmark.

Model selection used stratified five-fold cross-validation.

### 7.5 Hyperparameter Tuning

RandomizedSearchCV was used to optimize model configurations.

ROC-AUC was the primary search metric, while average precision and balanced accuracy were recorded as secondary metrics.

The selected Random Forest hyperparameters were:

| Parameter | Selected Value |
|---|---|
| n_estimators | 300 |
| max_depth | 10 |
| min_samples_split | 10 |
| min_samples_leaf | 4 |
| max_features | 0.7 |
| class_weight | balanced_subsample |
| random_state | 42 |

### 7.6 Probability Calibration

Sigmoid probability calibration was applied using CalibratedClassifierCV with three internal calibration folds.

Calibration was evaluated using Brier Score, Log Loss, ROC-AUC, average precision, and calibration curves.

The calibrated Random Forest was selected as the final research model because probability reliability improved while business-targeting performance remained strong.

## 8. Model Performance

### 8.1 Tuned Model Comparison

| Model | ROC-AUC | PR-AUC | Balanced Accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.8697 | 0.9725 | 0.6542 |
| Random Forest | **0.8811** | **0.9770** | **0.7193** |
| Gradient Boosting | 0.8759 | 0.9750 | 0.6328 |

Random Forest achieved the strongest tuned cross-validation results among the three algorithms.

These figures represent development-stage model-selection scores, not independent test results.

### 8.2 Probability Calibration Results

| Metric | Uncalibrated RF | Calibrated RF |
|---|---:|---:|
| Brier Score | 0.0917 | **0.0842** |
| Log Loss | 0.3249 | **0.2783** |
| ROC-AUC | **0.8778** | 0.8737 |
| PR-AUC | **0.9754** | 0.9710 |

Calibration reduced probability prediction error while producing relatively small decreases in ranking metrics.

### 8.3 Final Independent Test Results

The final calibrated Random Forest was evaluated using 240 held-out test customers.

| Metric | Test Score |
|---|---:|
| Accuracy | 0.8750 |
| Balanced Accuracy | 0.6091 |
| Precision | 0.8899 |
| Recall | 0.9758 |
| F1 Score | 0.9309 |
| ROC-AUC | **0.9157** |
| PR-AUC | **0.9851** |
| Brier Score | 0.0805 |
| Log Loss | 0.2539 |

The final model demonstrated strong risk-ranking performance.

However, retained-customer recall was only approximately 24% at the diagnostic classification threshold of 0.50.

Because the dataset is highly imbalanced, accuracy alone does not adequately represent predictive performance.

## 9. Model Explainability

Permutation importance was used to investigate global predictive feature importance.

The most influential features were:

| Feature | Importance |
|---|---:|
| monthly_usage_gb | 0.3454 |
| contract_type | 0.0382 |
| monthly_charges | 0.0170 |
| satisfaction_score | 0.0059 |
| tenure_months | 0.0058 |

Monthly usage was the dominant predictive feature.

Individual sensitivity analysis was also performed for three customers representing the highest, median, and lowest predicted churn risks in an internal validation subset.

These explanations describe model behavior and associations, not causal relationships.

## 10. Robustness Analysis

A feature-ablation experiment investigated the effect of removing monthly usage and its derived charge-to-usage ratio.

| Metric | Full Model | Without Usage |
|---|---:|---:|
| Validation ROC-AUC | 0.8811 | 0.6192 |
| PR-AUC | 0.9770 | 0.9071 |
| Balanced Accuracy | 0.7193 | 0.5180 |

Removing monthly usage reduced validation ROC-AUC by 0.2619.

This indicates that the model relies heavily on usage information.

The feature requires further validation because the dataset contains usage/service inconsistencies and lacks complete temporal information.

## 11. Customer Retention Simulation

The business scenario limits customer contact to the highest-risk 20%.

### Final Test Targeting Results

| Metric | Result |
|---|---:|
| Total Test Customers | 240 |
| Contact Capacity | 48 |
| Actual Churners | 207 |
| Churners Captured | **48** |
| Precision@20% | **1.0000** |
| Recall@20% | 0.2319 |
| Lift@20% | **1.1594** |
| Random Expected Captures | 41.4 |

The model identified all 48 actual churners among the 48 highest-risk test customers.

Random selection would capture approximately 41.4 churners in expectation.

The top-20% ranking rule is used to enforce the business contact limit, rather than relying on a universal probability threshold.

## 12. Financial Cost-Benefit Simulation

A hypothetical financial simulation was performed using development-data predictions.

### Business Assumptions

- Contact cost: 5 currency units
- Retained-customer value: 120 currency units
- Retention intervention success rate: 25%
- Customer contact capacity: 20%

### Simulation Results

| Metric | Random Selection | Model-Based |
|---|---:|---:|
| Churners Identified | 165.8 | 190 |
| Expected Customers Retained | 41.45 | 47.50 |
| Gross Benefit | 4,974 CU | 5,700 CU |
| Contact Cost | 960 CU | 960 CU |
| Net Benefit | 4,014 CU | **4,740 CU** |

**Estimated Incremental Net Benefit: 726 CU**

Sensitivity analysis produced incremental benefits ranging from 193.60 CU to 1,936 CU under different hypothetical assumptions.

These results demonstrate potential business value but are not verified financial returns.

## 13. Using the Saved Model

The final trained model is stored at:

```text
models/03_model_artifact.joblib
```

The complete saved artifact includes feature engineering, preprocessing, classification, and probability calibration.

Run the following Python code from the project root to verify that the model can generate predictions:

```python
from pathlib import Path
import pandas as pd
import joblib

ROOT = Path.cwd()

model = joblib.load(
    ROOT / "models" / "03_model_artifact.joblib"
)

customers = pd.read_csv(
    ROOT / "data" / "raw" /
    "customer_churn_advanced_dataset.csv"
).head(5)

X = customers.drop(
    columns=[
        "customer_id",
        "churn",
        "total_charges"
    ]
)

probabilities = model.predict_proba(X)[:, 1]

results = pd.DataFrame({
    "customer_id": customers["customer_id"],
    "predicted_churn_probability": probabilities
})

print(results)
```

This is an inference demonstration using existing dataset records, not an additional independent evaluation.

For a new retention batch, score all eligible customers, sort them by predicted churn probability in descending order, and select only the highest-risk 20% within the contact budget.

The saved model must only be loaded from trusted sources because Joblib deserialization can execute Python code.

## 14. Business Recommendations

Based on the predictive analysis, the following actions are recommended:

1. Prioritize customers according to predicted churn risk within the 20% contact-capacity limit.
2. Investigate tailored loyalty and renewal offers for high-risk month-to-month customers.
3. Strengthen complaint resolution and satisfaction monitoring.
4. Investigate service quality, pricing suitability, and customer expectations among high-usage customers.
5. Introduce human oversight, ongoing model monitoring, and controlled retention experiments.

Retention campaigns should test whether proposed interventions actually reduce churn rather than assuming predictive associations are causal.

## 15. Project Limitations

The project has several limitations:

- The dataset is highly imbalanced.
- Monthly usage is a dominant model feature.
- Some monthly usage records are inconsistent with internet-service categories.
- The dataset does not contain a complete temporal data dictionary.
- Potential target leakage cannot be fully ruled out.
- The final test dataset contains only 240 customers.
- The model exhibits some training-validation performance differences.
- Financial benefits rely on hypothetical assumptions.
- Predicting churn does not establish which customers are persuadable.
- Model performance on future customer populations has not been established.

The solution should therefore be treated as a research prototype.

Production deployment requires additional data validation, prospective testing, privacy review, and business intervention evaluation.

## 16. Reproducibility

The project uses a fixed random seed of 42.

Reproducibility is supported through:

- Versioned Python dependencies
- A stratified train-test split
- Stratified cross-validation
- Reusable Scikit-learn pipelines
- Documented hyperparameter searches
- Saved model metadata
- Exported evaluation results
- Saved visualizations
- An executable Jupyter Notebook

The package versions used to develop the model are available in `requirements.txt` and `models/model_metadata.json`.

Hyperparameter tuning and calibrated cross-validation can take several minutes to execute.


## 18. Conclusion

This project demonstrates an end-to-end machine learning workflow for predicting customer churn and supporting retention decisions.

The final calibrated Random Forest achieved a test ROC-AUC of 0.9157 and average precision of 0.9851.

Under the 20% contact-capacity constraint, the model identified 48 actual churners among the 48 highest-risk test customers.

A hypothetical financial simulation estimated 726 additional currency units in net benefit compared with random selection.

The results demonstrate the potential of machine learning for customer risk prioritization while emphasizing the importance of data quality, explainability, calibration, robustness, and human oversight.

**Final Recommendation:** Validate the reliability and timing of monthly usage data and conduct a controlled retention pilot before considering operational deployment.