HIGHLIGHTED_SKILLS = [
    "End-to-End Machine Learning Engineering",
    "Pipeline Design & Feature Engineering (Cyclic Encoding, One-Hot Encoding)",
    "Experiment Tracking & MLOps (MLflow)",
    "Ensemble & Gradient Boosting Architectures (CatBoost, XGBoost, LightGBM, Random Forest)",
    "Hyperparameter Optimization (RandomizedSearchCV)",
    "Model Interpretability (SHAP, Feature Importance)",
    "Robust Evaluation (Stratified K-Fold CV, Log-Transform Targets)"
]

 Bike Sharing Demand Predictive Pipeline & MLOps Experimentation

[![MLflow](https://img.shields.io/badge/MLops-MLflow-blue.svg)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

## Executive Summary
This project implements a rigorous, industry-grade machine learning pipeline designed to predict daily bike-sharing demand based on temporal, seasonal, and meteorological features. 

By leveraging advanced data preprocessing (such as cyclic encoding for temporal variables to preserve periodic relationships), implementing systematic cross-validation, and integrating **MLflow** for robust experiment tracking, this project highlights an engineering-first approach to predictive modeling. The final tuned **CatBoost** model achieved an outstanding **Test R² of 0.8891**, providing highly accurate and interpretable demand forecasts.

---

##  Key Technical Skills & Tools Used
- **Machine Learning Libraries:** `CatBoost`, `XGBoost`, `LightGBM`, `Scikit-Learn`
- **MLOps & Experiment Tracking:** `MLflow` (tracking parameters, custom metrics, and logging trusted model artifacts)
- **Feature Engineering:** Cyclic Sin/Cos Transformation, Outlier Detection, Multicollinearity Mitigation, Log-Transformation on targets
- **Model Interpretability:** `SHAP (SHapley Additive exPlanations)`, TreeExplainer, Global Feature Importance
- **Validation Strategy:** Custom Season-wise Stratified 80/20 Train-Test Split paired with a 5-Fold Cross-Validation strategy

---

##  The Pipeline Workflow

### 1. Exploratory Data Analysis & Multicollinearity Mitigation
- Analyzed distributions and target skewness.
- Removed `temp` due to an extremely high correlation (~0.99) with `atemp` to eliminate multicollinearity.
- Identified and parsed variables into distinct feature groups: cyclic categorical, binary, and continuous variables.

### 2. Feature Engineering & Preprocessing
- **Cyclic Temporal Encoding:** Transformed `mnth` and `weekday` features using Sine and Cosine transformations to capture the continuous nature of calendar periods (e.g., ensuring December is recognized as adjacent to January).
- **One-Hot Encoding:** Categorical variables (`season`, `weathersit`) were encoded dynamically.
- **Target Transformation:** Applied a log-transform ($y_{log} = \log(1 + y)$) to the target variable (`cnt`) to stabilize variance and normalize prediction residuals.
- **Scaling:** Features scaled using `StandardScaler` for distance-sensitive baseline models.

### 3. Model Pipeline & Comparison Strategy
We evaluated 8 baseline models on original-scale metrics using 5-Fold CV:
- **Linear Baselines:** Linear Regression, Ridge Regression
- **Tree-Based Models:** Decision Tree, Random Forest, Gradient Boosting Regressor
- **Advanced Boosting Regressors:** XGBoost, LightGBM, CatBoost

### 4. Hyperparameter Optimization & Fine-Tuning
Randomized hyperparameter searches (`RandomizedSearchCV`) were executed for the top-performing models (Random Forest, CatBoost, XGBoost, and LightGBM) to systematically boost regression scores.

---

## Model Performance Comparison

Below is the summary of our model performance progression from baseline to final tuned architectures:

| Model | Type | CV RMSE | Test RMSE | Test MAE | Test R² |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CatBoost** | **Tuned** | **659.64** | **618.07** | **440.42** | **0.8891** |
| **Random Forest** | **Tuned** | **695.73** | **622.16** | **424.55** | **0.8876** |
| **XGBoost** | **Tuned** | **690.02** | **623.82** | **451.06** | **0.8870** |
| **LightGBM** | **Tuned** | **747.46** | **713.82** | **515.23** | **0.8521** |
| *Random Forest* | *Baseline* | *708.22* | *625.40* | *430.95* | *0.8865* |
| *XGBoost* | *Baseline* | *757.49* | *630.95* | *451.85* | *0.8844* |
| *CatBoost* | *Baseline* | *689.78* | *646.63* | *458.94* | *0.8786* |

### Key Takeaways:
- **Tuned CatBoost** yielded the best balance, securing the lowest **Test RMSE of 618.07** and the highest **Test R² of 0.8891**.
- Tuning successfully reduced the generalization gap and improved CV stability across folds.

---

##  Model Interpretability & SHAP Analysis
To ensure the model is explainable for business stakeholders, we deployed **SHAP (SHapley Additive exPlanations)** to dissect the Tuned CatBoost model:
- **Primary Drivers:** Real-feel temperature (`atemp`) and year-over-year growth (`yr`) were the most powerful positive drivers of daily rentals.
- **Negative Constraints:** High humidity (`hum`), high windspeed, and adverse weather status (`weathersit_3`) negatively impacted rental counts, aligning cleanly with logical domain expectations.

---

## How to Run this Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Bike-Sharing-Demand-Predictive-Pipeline.git
   cd Bike-Sharing-Demand-Predictive-Pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run MLflow Dashboard:**
   ```bash
   mlflow ui
   ```
   Open `http://localhost:5000` in your browser to inspect the hyperparameter runs and artifact logs.
"""

print("=== SUGGESTED GITHUB REPOSITORY METADATA ===")
print(f"Project Name: {PROJECT_NAME}")
print(f"Project Description: {PROJECT_DESCRIPTION}\n")
print("=== HIGHLIGHTED SKILLS TO PUT ON RESUME/LINKEDIN ===")
for skill in HIGHLIGHTED_SKILLS:
    print(f"- {skill}")

# Save the README.md content directly to a file for easy download/push
with open("README.md", "w") as f:
    f.write(README_CONTENT)
print("\n[INFO] 'README.md' has been generated in your workspace directory!")
