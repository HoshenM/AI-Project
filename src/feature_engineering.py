import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def aggregate_feature_importance(best_models, output_csv="feature_importance_analysis.csv",
                                 output_fig="consolidated_feature_importance.png", top_n=15):
    """
    Aggregate and analyze feature importance across multiple models.

    Args:
        best_models (dict): Dictionary of trained models (pipelines).
        output_csv (str): Path to save feature importance analysis CSV.
        output_fig (str): Path to save consolidated feature importance plot.
        top_n (int): Number of top features to display.

    Returns:
        pd.DataFrame: DataFrame with average feature importance across models.
    """
    print("\n Feature Engineering Insights...")

    feature_importance_summary = {}

    for model_name in ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost']:
        if model_name in best_models:
            try:
                model = best_models[model_name].named_steps["model"]

                if hasattr(model, 'feature_importances_'):
                    feature_names = (best_models[model_name]
                                     .named_steps["preprocess"]
                                     .get_feature_names_out())

                    # Store feature importances as dict
                    importances = dict(zip(feature_names, model.feature_importances_))
                    feature_importance_summary[model_name] = importances

            except Exception as e:
                print(f"Could not extract feature importances for {model_name}: {str(e)}")

    # If we have results, aggregate
    if feature_importance_summary:
        all_features = set()
        for importances in feature_importance_summary.values():
            all_features.update(importances.keys())

        feature_importance_df = pd.DataFrame()
        for feature in all_features:
            feature_scores = {}
            for model_name, importances in feature_importance_summary.items():
                feature_scores[model_name] = importances.get(feature, 0)
            feature_importance_df[feature] = feature_scores

        feature_importance_df = feature_importance_df.T
        feature_importance_df['Average_Importance'] = feature_importance_df.mean(axis=1)
        feature_importance_df = feature_importance_df.sort_values('Average_Importance', ascending=False)

        # Print summary
        print("\n Top Features (Average Importance across models) ")
        print(feature_importance_df.head(top_n).round(4))

        # Save to CSV
        feature_importance_df.to_csv(output_csv)
        print(f"Feature importance analysis saved as: {output_csv}")

        # Plot top features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['Average_Importance'])
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Average Feature Importance')
        plt.title(f'Top {top_n} Features - Average Importance Across Models', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_fig, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Consolidated feature importance plot saved as: {output_fig}")

        return feature_importance_df

    else:
        print("No feature importance data available.")
        return pd.DataFrame()
