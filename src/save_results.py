# src/save_results.py

import json
import joblib
import numpy as np

def convert_numpy(obj):
    """Helper function to convert numpy types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):  # recursive for dictionaries
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):  # recursive for lists
        return [convert_numpy(item) for item in obj]
    return obj


def save_results(best_model_final, best_model_name, df, X, y, X_train, X_valid, X_test,
                 models, results_df, results_val, test_results,
                 cv_results, overfitting_analysis, runtime_results):
    """
    Save best model and comprehensive results to disk.
    
    Args:
        best_model_final: trained best model
        best_model_name (str): name of the best model
        df (pd.DataFrame): original dataset
        X (pd.DataFrame): features
        y (pd.Series): target
        X_train, X_valid, X_test: train/valid/test features
        models (dict): dictionary of models
        results_df (pd.DataFrame): validation results
        results_val (dict): validation metrics per model
        test_results (dict): test metrics per model
        cv_results (dict): cross-validation results
        overfitting_analysis (dict): overfitting gap analysis
        runtime_results (dict): runtime statistics
    """

    print("\n Saving Best Model and Results...")

    # Save best model
    model_filename = f"best_stroke_model_{best_model_name}.pkl"
    joblib.dump(best_model_final, model_filename)
    print(f"Best model saved as: {model_filename}")

    # Prepare comprehensive report
    final_report = {
        'dataset_info': {
            'source': 'https://www.kaggle.com/shriyashjagtap/stroke-diagnosis-and-health-metrics-data',
            'total_samples': len(df),
            'features': len(X.columns),
            'target_distribution': y.value_counts().to_dict(),
        },
        'data_splits': {
            'train_size': len(X_train),
            'validation_size': len(X_valid),
            'test_size': len(X_test),
        },
        'models_evaluated': list(models.keys()),
        'best_model': {
            'name': best_model_name,
            'validation_auc': results_df.loc[best_model_name, 'AUC'],
            'test_auc': test_results[best_model_name]['Test_AUC'],
            'best_parameters': results_val[best_model_name]['Best_Params']
        },
        'all_validation_results': results_val,
        'all_test_results': test_results,
        'cross_validation_results': {k: v for k, v in cv_results.items() if k != 'CV_Scores'},
        'overfitting_analysis': overfitting_analysis,
        'training_times': runtime_results
    }

    # Save JSON
    with open("comprehensive_results.json", "w") as f:
        json.dump(convert_numpy(final_report), f, indent=2)

    print("Comprehensive results saved as: comprehensive_results.json")
