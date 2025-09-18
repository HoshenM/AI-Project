import numpy as np
from scipy import stats


def compare_models_statistically(scores1, scores2, model1_name, model2_name, alpha=0.05):
    """
    Compare two models using a paired t-test on cross-validation scores.

    Args:
        scores1 (array-like): Cross-validation scores of model 1.
        scores2 (array-like): Cross-validation scores of model 2.
        model1_name (str): Name of model 1.
        model2_name (str): Name of model 2.
        alpha (float): Significance level (default 0.05).

    Returns:
        dict: Results containing t-statistic, p-value, and interpretation.
    """
    statistic, p_value = stats.ttest_rel(scores1, scores2)

    result = {
        "Model_1": model1_name,
        "Model_2": model2_name,
        "T-statistic": statistic,
        "P-value": p_value,
        "Significance_Level": alpha,
        "Conclusion": None
    }

    if p_value < alpha:
        better_model = model1_name if np.mean(scores1) > np.mean(scores2) else model2_name
        result["Conclusion"] = f"Statistically significant difference. {better_model} performs better."
    else:
        result["Conclusion"] = "No statistically significant difference."

    print(f"\nComparing {model1_name} vs {model2_name}:")
    print(f"T-statistic: {statistic:.4f}, P-value: {p_value:.4f}")
    print(result["Conclusion"])

    return result
