import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay,
    PrecisionRecallDisplay, classification_report
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

from src.hardware_monitor import monitor_resource_usage


def train_and_evaluate(models, param_grids, preprocessor,
                       X_train, y_train, X_valid, y_valid, pp):
    """
    Train and evaluate models using GridSearchCV, 
    return best models and evaluation results.
    """

    best_models = {}
    results_val = {}
    cv_results = {}
    runtime_results = {}
    overfitting_analysis = {}
    resource_usage = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n Optimizing {name}...")

        # Pipeline
        pipe = ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTETomek(random_state=42)),
            ("model", model)
        ])

        # Monitor resource usage
        start_time = time.time()
        cpu_before, mem_before = monitor_resource_usage()

        grid = GridSearchCV(
            pipe, param_grids[name], cv=cv, scoring="roc_auc",
            n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)

        training_time = time.time() - start_time
        cpu_after, mem_after = monitor_resource_usage()

        runtime_results[name] = round(training_time, 2)
        resource_usage[name] = {
            "CPU_Usage": cpu_after - cpu_before,
            "Memory_Usage": mem_after - mem_before
        }

        # Save best model
        best_pipe = grid.best_estimator_
        best_models[name] = best_pipe

        # Predictions
        val_preds = best_pipe.predict(X_valid)
        val_proba = best_pipe.predict_proba(X_valid)[:, 1]

        # Metrics
        val_acc = accuracy_score(y_valid, val_preds)
        val_auc = roc_auc_score(y_valid, val_proba)
        val_f1 = f1_score(y_valid, val_preds)
        val_precision = precision_score(y_valid, val_preds)
        val_recall = recall_score(y_valid, val_preds)

        cv_scores = cross_val_score(best_pipe, X_train, y_train,
                                    cv=5, scoring='roc_auc')

        results_val[name] = {
            "Accuracy": val_acc,
            "AUC": val_auc,
            "F1": val_f1,
            "Precision": val_precision,
            "Recall": val_recall,
            "Best_Params": grid.best_params_,
            "Train_Time_s": training_time
        }

        cv_results[name] = {
            "CV_Mean": cv_scores.mean(),
            "CV_Std": cv_scores.std(),
            "CV_Scores": cv_scores
        }

        # Overfitting analysis
        train_proba = best_pipe.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)
        overfitting_analysis[name] = {
            "Train_AUC": train_auc,
            "Valid_AUC": val_auc,
            "Overfitting_Gap": train_auc - val_auc
        }

        print(f" {name} - Validation AUC: {val_auc:.4f}, Training time: {training_time:.2f}s")

        # Visualizations 
        cm = confusion_matrix(y_valid, val_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Stroke", "Stroke"],
                    yticklabels=["No Stroke", "Stroke"])
        plt.title(f"{name} - Confusion Matrix", fontweight='bold')
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        pp.savefig(bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 6))
        RocCurveDisplay.from_predictions(y_valid, val_proba)
        plt.title(f"{name} - ROC Curve (AUC = {val_auc:.4f})", fontweight='bold')
        plt.grid(True, alpha=0.3)
        pp.savefig(bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 6))
        PrecisionRecallDisplay.from_predictions(y_valid, val_proba)
        plt.title(f"{name} - Precision-Recall Curve", fontweight='bold')
        plt.grid(True, alpha=0.3)
        pp.savefig(bbox_inches='tight')
        plt.close()

    return best_models, results_val, cv_results, runtime_results, overfitting_analysis, resource_usage
