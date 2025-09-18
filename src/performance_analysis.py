import pandas as pd
import numpy as np

def compare_validation_test_performance(results_val, test_results, output_csv="detailed_performance_comparison.csv"):
    """
    Compare validation vs. test performance across models.

    Args:
        results_val (dict): Validation results for each model (metrics stored in dicts).
        test_results (dict): Test results for each model (metrics stored in dicts).
        output_csv (str): Path to save the performance comparison CSV.

    Returns:
        pd.DataFrame: Performance comparison DataFrame.
    """
    print("\n Detailed Performance Analysis...")

    performance_metrics = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall']
    performance_comparison = pd.DataFrame()

    for model_name in results_val.keys():
        # Ensure model_name exists in test_results
        if model_name in test_results:
            model_performance = {}
            for metric in performance_metrics:
                val_metric = results_val[model_name].get(metric, 0)
                test_metric = test_results[model_name].get(f'Test_{metric}', 0)

                model_performance[f'Validation_{metric}'] = val_metric
                model_performance[f'Test_{metric}'] = test_metric

                # Calculate absolute difference if both values are numeric
                if isinstance(val_metric, (int, float)) and isinstance(test_metric, (int, float)):
                    model_performance[f'{metric}_Difference'] = abs(val_metric - test_metric)
                else:
                    model_performance[f'{metric}_Difference'] = np.nan

            performance_comparison[model_name] = model_performance

    # Transpose for easier readability
    performance_comparison = performance_comparison.T

    print("\n Validation vs Test Performance Comparison")
    print(performance_comparison.round(4))

    # Save to CSV
    performance_comparison.to_csv(output_csv)
    print(f"Detailed performance comparison saved as: {output_csv}")

    return performance_comparison

