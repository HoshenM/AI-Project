import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_importance
import seaborn as sns


def plot_feature_importances(best_models, importance_models, pp, top_n=15):
    """
    Plot feature importances for selected models.

    Args:
        best_models (dict): Dictionary of trained pipelines (model name -> pipeline).
        importance_models (list): Models to compute feature importances for.
        pp (PdfPages): PdfPages object to save figures.
        top_n (int): Number of top features to plot.

    Returns:
        None
    """

    print("\n Analyzing Feature Importance...")

    for model_name in importance_models:
        if model_name not in best_models:
            continue

        print(f"Processing feature importance for {model_name}...")

        # Extract trained model (after preprocessing)
        model = best_models[model_name].named_steps["model"]

        # --- Tree-based models with feature_importances_ ---
        if hasattr(model, "feature_importances_"):
            preprocessed_features = (
                best_models[model_name]
                .named_steps["preprocess"]
                .get_feature_names_out()
            )

            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]

            plt.figure(figsize=(10, 8))
            plt.barh(
                range(len(indices)),
                importances[indices],
                align="center",
                color="skyblue"
            )
            plt.yticks(
                range(len(indices)),
                [preprocessed_features[i].replace("cat__", "").replace("num__", "")
                 for i in indices]
            )
            plt.xlabel("Feature Importance")
            plt.title(f"Top {top_n} Feature Importances ({model_name})", fontweight="bold")
            plt.gca().invert_yaxis()
            pp.savefig(bbox_inches="tight"); plt.close()

        # --- XGBoost built-in ---
        elif model_name == "XGBoost":
            plt.figure(figsize=(10, 8))
            plot_importance(model, max_num_features=top_n)
            plt.title(f"Top {top_n} Feature Importances ({model_name})", fontweight="bold")
            plt.tight_layout()
            pp.savefig(bbox_inches="tight"); plt.close()

        # --- CatBoost built-in ---
        elif model_name == "CatBoost":
            feature_importances = model.get_feature_importance()
            feature_names = (
                best_models[model_name]
                .named_steps["preprocess"]
                .get_feature_names_out()
            )
            sorted_idx = np.argsort(feature_importances)[::-1][:top_n]

            plt.figure(figsize=(10, 8))
            sns.barplot(
                x=feature_importances[sorted_idx],
                y=[feature_names[i].replace("cat__", "").replace("num__", "")
                   for i in sorted_idx],
                orient="h"
            )
            plt.xlabel("Feature Importance")
            plt.title(f"Top {top_n} Feature Importances ({model_name})", fontweight="bold")
            plt.tight_layout()
            pp.savefig(bbox_inches="tight"); plt.close()
