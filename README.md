# Introduction to Artificial Intelligence – Final Project  
**Stroke Prediction using AI & Machine Learning**

This project applies **exploratory data analysis (EDA)**, **data preprocessing**, and both **supervised** and **ensemble learning methods** to a healthcare dataset.

It explores **risk factors for stroke** through feature analysis, evaluates **14 machine learning models** with cross-validation and hyperparameter tuning, and validates findings with **statistical significance tests**.  

All outputs are exported as **publication-ready figures and reports** into the `results/` and `figures/` directories.

---

## Project Structure
```bash
stroke-classification-project/
│
├── data/                         # Raw and processed datasets
│   └── stroke_data.csv
│
├── figures/                      # All generated visualizations & reports
│   ├── eda_report.pdf
│   ├── stroke_project_report.pdf
│   ├── validation_curve_*.png
│   └── consolidated_feature_importance.png
│
├── results/                      # Saved results and reports
│   ├── model_comparison.csv
│   ├── cv_results.csv
│   ├── final_test_results.csv
│   ├── comprehensive_results.json
│   └── feature_importance_analysis.csv
│
├── models/                       # Serialized trained models & functions
│   ├── best_stroke_model_<NAME>.pkl
│   └── stroke_prediction_function.pkl
│
├── src/                          # Source code (modular pipeline)
│   ├── data_loader.py            # Load dataset
│   ├── eda_analysis.py           # Exploratory Data Analysis
│   ├── preprocessing.py          # Data preprocessing & feature encoding
│   ├── model_setup.py            # Define models & hyperparameter grids
│   ├── model_training.py         # Training, tuning & evaluation
│   ├── results_analysis.py       # Analyze validation & CV results
│   ├── feature_importance.py     # Plot feature importances
│   ├── ensemble_methods.py       # Voting ensemble creation
│   ├── learning_curves.py        # Generate learning curves
│   ├── evaluation.py             # Final test set evaluation
│   ├── statistical_tests.py      # Statistical significance testing
│   ├── project_summary.py        # Print project summary
│   ├── save_results.py           # Save best model & full results
│   ├── validation_curves.py      # Validation curves for hyperparameters
│   ├── performance_analysis.py   # Compare validation vs test performance
│   ├── feature_engineering.py    # Aggregate feature importance insights
│   ├── deployment.py             # Deployment functions (prediction API)
│   ├── final_report.py           # Final project report
│   └── hardware_monitor.py       # Track CPU / memory usage
│
├── main.py                       # Orchestrator script (pipeline entrypoint)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation


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

## Outputs (PDFs in `figures_pdf/`)

- **fig1.pdf** — Feature distributions (EDA)  
- **fig2.pdf** — Correlation heatmap  
- **fig3.pdf** — Model comparison & selection  
- **fig4.pdf** — Confusion matrices, ROC & PR curves  
- **fig5.pdf** — Ensemble results & statistical tests  
- **final_report.pdf** — Complete project report

---

## Key Results (current run)

- **Best Individual Model**: Logistic Regression (L1)  
  - Accuracy: **0.842**  
  - ROC-AUC: **0.856**  
  - Recall (Stroke class): **0.781**

- **Best Ensemble**: Soft Voting (LR + Random Forest + CatBoost)  
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

- **preprocessing.py** — cleaning, encoding, scaling

- **eda_analysis.py** — EDA plots & feature correlations

- **model_training.py** — model wrappers & training functions

- **hyperparameter_tuning.py** — GridSearchCV & Optuna optimization

- **evaluation.py** — metrics, curves, confusion matrices

- **ensemble_methods.py** — voting, bagging, boosting

- **pdf_saver.py** — saves figures as PDFs

- **main.py** — orchestrates entire pipeline

---

## Data & Code

- Dataset: [Stroke Diagnosis and Health Metrics Data](https://www.kaggle.com/datasets/shriyashjagtap/stroke-diagnosis-and-health-metrics-data)


- This repo includes all scripts to reproduce the analysis and figures.

---

## Contact

Questions or feedback: hoshenmn@gmail.com
