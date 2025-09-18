# src/project_summary.py

import pandas as pd

def summarize_project(models, best_model_name, results_df, test_results,
                      runtime_results, overfitting_df, pdf_saver):
    """
    Print and save a project summary with recommendations.
    Closes the open PDF saver at the end.

    Args:
        models (dict): Dictionary of trained models
        best_model_name (str): Name of best-performing model
        results_df (pd.DataFrame): Validation results
        test_results (dict): Final test set results
        runtime_results (dict): Training runtimes
        overfitting_df (pd.DataFrame): Overfitting analysis
        pdf_saver: PdfPages object used for saving plots
    """
    print("\n Project Summary")
    print("=" * 50)
    print(f"Total models evaluated: {len(models)}")
    print(f"Best performing model: {best_model_name}")
    print(f"Best validation AUC: {results_df.loc[best_model_name, 'AUC']:.4f}")
    print(f"Best test AUC: {test_results[best_model_name]['Test_AUC']:.4f}")
    print(f"Total training time: {sum(runtime_results.values()):.2f} seconds")

    # Recommendations
    print("\n Recommendations:")
    print(f"1. Best overall model: {best_model_name}")
    print(f"2. Fastest model: {min(runtime_results, key=runtime_results.get)}")
    print(f"3. Most balanced model (F1): {results_df['F1'].idxmax()}")

    # Overfitting concerns
    high_overfitting = overfitting_df[overfitting_df['Overfitting_Gap'] > 0.05]
    if not high_overfitting.empty:
        print(f"4. Models with potential overfitting: {list(high_overfitting.index)}")
    else:
        print("4. No significant overfitting detected in any model")

    # Close PDF
    pdf_saver.close()
    print("\n All visualizations saved to: stroke_project_results.pdf")

