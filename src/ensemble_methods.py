import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score


def build_voting_ensemble(best_models, results_df, X_train, y_train, X_valid, y_valid):
    """
    Build and evaluate a soft voting classifier using top models by AUC.

    Args:
        best_models (dict): Dictionary of trained pipelines (model name -> pipeline).
        results_df (pd.DataFrame): Validation results DataFrame with an "AUC" column.
        X_train, y_train: Training data.
        X_valid, y_valid: Validation data.

    Returns:
        tuple: (best_models, results_val_update, voting_auc)
    """

    print("\n Creating Ensemble Models...")

    # Select top 3 models by validation AUC
    top_models = results_df.nlargest(3, "AUC")
    print(f"Top 3 models for ensemble: {list(top_models.index)}")

    if len(top_models) < 3:
        print("Not enough models for an ensemble.")
        return best_models, {}, None

    # Create soft voting classifier
    voting_estimators = [(name, best_models[name]) for name in top_models.index[:3]]
    voting_clf = VotingClassifier(voting_estimators, voting="soft")
    voting_clf.fit(X_train, y_train)

    # Evaluate on validation set
    voting_val_proba = voting_clf.predict_proba(X_valid)[:, 1]
    voting_val_auc = roc_auc_score(y_valid, voting_val_proba)

    print(f"Voting Classifier Validation AUC: {voting_val_auc:.4f}")

    # Update dictionaries
    best_models["VotingClassifier"] = voting_clf
    results_val_update = {
        "VotingClassifier": {
            "AUC": voting_val_auc,
            "Ensemble_of": list(top_models.index[:3])
        }
    }

    return best_models, results_val_update, voting_val_auc
