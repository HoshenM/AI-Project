# src/validation_curves.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

def plot_validation_curve_analysis(model, X, y, param_name, param_range, model_name):
    """
    Plot validation curve for a specific parameter.

    Args:
        model: pipeline (with preprocessing + model)
        X (pd.DataFrame): training features
        y (pd.Series): training target
        param_name (str): hyperparameter name (e.g., 'model__n_estimators')
        param_range (list): list of values for the parameter
        model_name (str): name of the model
    """
    try:
        train_scores, valid_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=3, scoring="roc_auc", n_jobs=-1
        )

        plt.figure(figsize=(10, 6))
        plt.plot(param_range, np.mean(train_scores, axis=1), "o-", label="Training", color="blue")
        plt.plot(param_range, np.mean(valid_scores, axis=1), "o-", label="Validation", color="red")

        plt.fill_between(param_range,
                         np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                         np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                         alpha=0.2, color="blue")
        plt.fill_between(param_range,
                         np.mean(valid_scores, axis=1) - np.std(valid_scores, axis=1),
                         np.mean(valid_scores, axis=1) + np.std(valid_scores, axis=1),
                         alpha=0.2, color="red")

        plt.xlabel(param_name)
        plt.ylabel("AUC Score")
        plt.title(f"Validation Curve - {model_name} - {param_name}", fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)

        filename = f"validation_curve_{model_name}_{param_name.replace('__', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Validation curve saved for {model_name} - {param_name} -> {filename}")

    except Exception as e:
        print(f"Could not generate validation curve for {model_name} - {param_name}: {str(e)}")


def run_validation_curves(models, X_train, y_train, preprocessor):
    """
    Generate validation curves for selected models and parameters.

    Args:
        models (dict): dictionary of models
        X_train: training features
        y_train: training target
        preprocessor: ColumnTransformer
    """
    validation_curve_configs = {
        "RandomForest": [
            ("model__n_estimators", [50, 100, 150, 200, 250, 300]),
            ("model__max_depth", [3, 5, 7, 10, 15, None])
        ],
        "XGBoost": [
            ("model__n_estimators", [50, 100, 150, 200, 250, 300]),
            ("model__learning_rate", [0.01, 0.05, 0.1, 0.15, 0.2, 0.3])
        ],
        "LogisticRegression": [
            ("model__C", [0.001, 0.01, 0.1, 1, 10, 100, 1000])
        ]
    }

    for model_name, param_configs in validation_curve_configs.items():
        if model_name in models:
            base_pipe = ImbPipeline([
                ("preprocess", preprocessor),
                ("smote", SMOTETomek(random_state=42)),
                ("model", models[model_name])
            ])

            for param_name, param_range in param_configs:
                plot_validation_curve_analysis(base_pipe, X_train, y_train,
                                               param_name, param_range, model_name)
