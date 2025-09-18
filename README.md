# Introduction to Artificial Intelligence – Final Project  
**Stroke Prediction using AI & Machine Learning**

This project applies **exploratory data analysis (EDA)**, **data preprocessing**, and both **supervised** and **ensemble learning methods** to a healthcare dataset.

It explores **risk factors for stroke** through feature analysis, evaluates **14 machine learning models** with cross-validation and hyperparameter tuning, and validates findings with **statistical significance tests**.  

All outputs are exported as **publication-ready figures and reports** into the `results/` and `figures/` directories.

---

## Project Structure
```bash
AI-Project/
│
├── data/                         # Raw dataset
│   └── stroke_data.csv
│
├── figures/                      # Generated visualizations & reports
│   ├── eda_report.pdf
│   ├── stroke_project_result.pdf
│   ├── validation_curve_XGBoost_model_n_estimators.png
│   ├── validation_curve_XGBoost_model_learning_rate.png
│   ├── validation_curve_LogisticRegression_model_C.png
│   ├── validation_curve_XGBoost_model_n_estimators.png
│   ├── validation_curve_RandomForest_model_n_estimators.png
│   ├── consolidated_feature_importance.png
|   └── Llearninig curve- logisticregrussion_L1.png
│
├── results/                      # Saved results and reports
│   ├── model_comparison.csv
│   ├── cv_results.csv
│   ├── final_test_results.csv
│   ├── comprehensive_results.json
│   ├── feature_importance_analysis.csv
│   ├── overfitting_analysis.csv
│   └── detailed_performance_comparison.csv
│
├── models/                       # Serialized trained models
│   ├── best_stroke_model_LogisticRegression_L1.pkl
│   └── stroke_prediction_function.pkl
│
├── src/                          # Source code (pipeline modules)
│   ├── data_loader.py
│   ├── eda_analysis.py
│   ├── preprocessing.py
│   ├── model_setup.py
│   ├── model_training.py
│   ├── results_analysis.py
│   ├── feature_importance.py
│   ├── ensemble_methods.py
│   ├── learning_curves.py
│   ├── evaluation.py
│   ├── statistical_tests.py
│   ├── project_summary.py
│   ├── save_results.py
│   ├── validation_curves.py
│   ├── performance_analysis.py
│   ├── feature_engineering.py
│   ├── deployment.py
│   ├── final_report.py
│   └── hardware_monitor.py
│
├── main.py                       # Pipeline entrypoint
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation                  # Project documentation


```
---
## Methods Used

### Preprocessing & EDA
- Dropped missing values and inconsistent rows.  
- Standardized continuous variables (**Age**, **BMI**, **Avg_Glucose**).  
- One-hot encoded categorical variables (**Gender**, **SES**, **Smoking_Status**).  
- **EDA outputs**:  
  - **fig1.pdf**: feature distributions.  
  - **fig2.pdf**: correlation heatmap & target associations.  

### Data Splitting & Scaling
- **Stratified split**: 70% train, 15% validation, 15% test.  
- **StandardScaler** fit only on train set, applied to val/test.  
- Validation used for **early stopping** (when supported).  

### Models
Evaluated **14 supervised models**, including:  
- Logistic Regression (L1, L2)  
- Decision Tree, Random Forest  
- Gradient Boosting, XGBoost, LightGBM, CatBoost  
- SVM (linear, RBF)  
- KNN, Naive Bayes  
- Ensemble methods (Voting, Bagging, Boosting)  

### Hyperparameter Tuning
- Used **GridSearchCV** and **Optuna** (5-fold time-aware CV).  
- Optimized depth, learning rate, regularization, number of estimators, and class weights.  

### Evaluation & Selection
- Metrics: **Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC**.  
- Figures:  
  - **fig3.pdf**: model comparison & ranking.  
  - **fig4.pdf**: confusion matrices, ROC & PR curves.  

### Statistical Validation
- Applied **significance tests** (paired t-test, McNemar’s test) to compare classifiers.  
- Highlighted differences in recall/precision for minority class (**stroke cases**).  

### Ensemble Methods
- Implemented **Soft Voting Classifier** combining best 3 models.  
- Ensemble improved overall AUC and recall compared to individual models.  
- **fig5.pdf**: ensemble performance.  

---

## Setup

### 1) Create a virtual environment (optional)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2) Install dependencies
If you have `requirements.txt`:
```bash
pip install -r requirements.txt
```
Otherwise:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost optuna reportlab pypdf
```

### 3) Dataset
Place `stroke_data.csv` inside the `data/` folder.

---

## How to Run
Run the full pipeline:
```bash
python main.py
```
This will:

1. Preprocess data and run EDA (**fig1.pdf, fig2.pdf**)

2. Train & tune 14 supervised models

3. Generate evaluation reports (**fig3.pdf, fig4.pdf**)

4. Train ensemble model and save results (**fig5.pdf**)

5. Export final PDF report to `results/`

---

## Outputs (in figures/)

- eda_report.pdf — Exploratory Data Analysis report

- stroke_project_report.pdf — Combined model evaluation report

- validation_curve_*.png — Validation curves for selected models

- consolidated_feature_importance.png — Top features across models

- final_report.pdf — Complete project summary
---

## Key Results (latest run)

- **Best Individual Model**: Logistic Regression (L1)  
  - Accuracy: **0.842**  
  - ROC-AUC: **0.856** (primary evaluation metric for imbalanced data)  
  - Recall (Stroke class): **0.781**

- **Best Ensemble**: Soft Voting (Logistic Regression + Random Forest + CatBoost)  
  - Accuracy: **0.849**  
  - ROC-AUC: **0.869**  
  - Recall (Stroke class): **0.804**

---

## Reproducibility

- `random_state=42` everywhere

- **Stratified splits** for balanced folds

- **Scaling** applied only after splitting

---

## Files — What They Do

- data_loader.py — loads the dataset from `data/`

- eda_analysis.py — generates EDA plots & correlation heatmaps

- preprocessing.py — data cleaning, feature engineering, encoding, and scaling

- model_setup.py — defines ML models and hyperparameter grids

- model_training.py — trains models, runs cross-validation, hyperparameter tuning

- results_analysis.py — summarizes validation, cross-validation, and overfitting results

- feature_importance.py — plots feature importances for tree-based models

- ensemble_methods.py — builds and evaluates ensemble models (e.g. voting classifier)

- learning_curves.py — generates learning curves for top models

- evaluation.py — evaluates final model(s) on the test set

- statistical_tests.py — runs statistical significance tests between models

- project_summary.py — prints project summary and recommendations

- save_results.py — saves trained models and results to disk (`results/` + `models/`)

- validation_curves.py — generates validation curves for key hyperparameters

- performance_analysis.py — compares validation vs test metrics

- feature_engineering.py — analyzes and aggregates feature importance across models

- deployment.py — provides prediction API for new patient data + saves predictor function

- final_report.py — prints the final report with outputs and findings

- hardware_monitor.py — tracks CPU and memory usage during training

- main.py — orchestrates the full pipeline (entrypoint)

- requirements.txt — list of dependencies

---

## Data & Code

- Dataset: [Stroke Diagnosis and Health Metrics Data](https://www.kaggle.com/datasets/shriyashjagtap/stroke-diagnosis-and-health-metrics-data)


- This repo includes all scripts to reproduce the analysis and figures.

---

## Contact

Questions or feedback: hoshenmn@gmail.com
