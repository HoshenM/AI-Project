import pandas as pd


def analyze_results(results_val, cv_results, overfitting_analysis,
                    results_path="model_comparison.csv",
                    cv_path="cv_results.csv",
                    overfit_path="overfitting_analysis.csv"):
    """
    Analyze model performance results, cross-validation, and overfitting.

    Args:
        results_val (dict): Validation metrics from training.
        cv_results (dict): Cross-validation scores.
        overfitting_analysis (dict): Train vs validation AUC gaps.
        results_path (str): Path to save validation results CSV.
        cv_path (str): Path to save CV results CSV.
        overfit_path (str): Path to save overfitting analysis CSV.

    Returns:
        results_df (pd.DataFrame): Validation results dataframe.
        cv_df (pd.DataFrame): CV results dataframe.
        overfitting_df (pd.DataFrame): Overfitting dataframe.
    """

    print("\n Analyzing Results...")

    # Convert dictionaries to DataFrames
    results_df = pd.DataFrame(results_val).T
    results_df['AUC'] = pd.to_numeric(results_df['AUC'], errors='coerce')

    cv_df = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != 'CV_Scores'}
                          for k, v in cv_results.items()}).T
    overfitting_df = pd.DataFrame(overfitting_analysis).T

    # Print summaries
    print("\n Validation Results ")
    print(results_df.round(4))

    print("\n Cross-Validation Results ")
    print(cv_df.round(4))

    print("\n Overfitting Analysis ")
    print(overfitting_df.round(4))

    # Save to CSV
    results_df.to_csv(results_path)
    cv_df.to_csv(cv_path)
    overfitting_df.to_csv(overfit_path)

    return results_df, cv_df, overfitting_df
