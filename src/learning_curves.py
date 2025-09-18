import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_learning_curves(best_model, best_model_name, X_train, y_train, pdf_saver=None):
    """
    Generate and save learning curves for the best model.

    Args:
        best_model: Trained pipeline (Pipeline or VotingClassifier).
        best_model_name (str): Name of the best model.
        X_train, y_train: Training data.
        pdf_saver (PdfPages or None): If provided, saves figure to this PDF.
    """

    print(f"\n Generating learning curves for best model: {best_model_name}")

    # Compute learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        best_model,
        X_train, y_train,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="roc_auc",
        n_jobs=-1
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="Training Score")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), "o-", label="Cross-Validation Score")

    # Fill bands
    plt.fill_between(train_sizes,
                     np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                     alpha=0.1)
    plt.fill_between(train_sizes,
                     np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                     alpha=0.1)

    plt.xlabel("Training Set Size")
    plt.ylabel("AUC Score")
    plt.title(f"Learning Curves - {best_model_name}", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save or show
    if pdf_saver:
        pdf_saver.savefig(bbox_inches="tight")
        plt.close()
    else:
        plt.show()
