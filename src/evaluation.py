import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)


def evaluate_on_test_set(best_models, best_model_name, X_test, y_test, pdf_saver=None):
    """
    Evaluate all trained models on the test set.
    Produces metrics, confusion matrix, and classification report.

    Args:
        best_models (dict): Dictionary of fitted models.
        best_model_name (str): Name of the best-performing model.
        X_test, y_test: Test data.
        pdf_saver (PdfPages or None): If provided, saves plots into PDF.

    Returns:
        final_test_df (pd.DataFrame): DataFrame with test metrics for each model.
    """

    print("\n Final Test Set Evaluation...")
    test_results = {}

    # Evaluate each model (skip VotingClassifier for simplicity)
    for name, model in best_models.items():
        if name == "VotingClassifier":
            continue

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        test_results[name] = {
            "Test_Accuracy": accuracy_score(y_test, preds),
            "Test_AUC": roc_auc_score(y_test, proba),
            "Test_F1": f1_score(y_test, preds),
            "Test_Precision": precision_score(y_test, preds),
            "Test_Recall": recall_score(y_test, preds)
        }

    final_test_df = pd.DataFrame(test_results).T
    print("\n Final Test Results ")
    print(final_test_df.round(4))

    # Save results
    final_test_df.to_csv("final_test_results.csv")

    # Detailed evaluation for the best model
    best_model_final = best_models[best_model_name]
    final_preds = best_model_final.predict(X_test)
    final_proba = best_model_final.predict_proba(X_test)[:, 1]

    print(f"\n FINAL RESULTS - Best Model: {best_model_name}")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy_score(y_test, final_preds):.4f}")
    print(f"Test AUC: {roc_auc_score(y_test, final_proba):.4f}")
    print(f"Test F1: {f1_score(y_test, final_preds):.4f}")
    print(f"Test Precision: {precision_score(y_test, final_preds):.4f}")
    print(f"Test Recall: {recall_score(y_test, final_preds):.4f}")
    print("=" * 60)

    # Confusion matrix for the best model
    cm_final = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_final, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Stroke", "Stroke"],
                yticklabels=["No Stroke", "Stroke"])
    plt.title(f"Final Test Confusion Matrix - {best_model_name}", fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    if pdf_saver:
        pdf_saver.savefig(bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # Classification report
    print("\n Detailed Classification Report:")
    print(classification_report(y_test, final_preds, target_names=["No Stroke", "Stroke"]))

    return final_test_df
