from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_models(pos_weight):
    """Return dictionary of ML models for stroke prediction."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "LogisticRegression_L1": LogisticRegression(penalty='l1', solver='liblinear', class_weight="balanced", random_state=42),
        "LogisticRegression_L2": LogisticRegression(penalty='l2', class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "RandomForest_Regularized": RandomForestClassifier(max_depth=5, min_samples_split=20, class_weight="balanced", random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight, use_label_encoder=False, random_state=42),
        "LightGBM": LGBMClassifier(scale_pos_weight=pos_weight, random_state=42, verbosity=-1),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(max_iter=500, early_stopping=True, random_state=42),
        "MLP_Regularized": MLPClassifier(max_iter=500, early_stopping=True, validation_fraction=0.1, alpha=0.01, random_state=42),
        "CatBoost": CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, eval_metric="AUC", verbose=0, random_state=42),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=42)
    }
    return models

def get_param_grids():
    """Return hyperparameter grids for all models."""
    param_grids = {
        "LogisticRegression": {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]},
        "LogisticRegression_L1": {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]},
        "LogisticRegression_L2": {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]},
        "RandomForest": {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 5, 10, 15]},
        "RandomForest_Regularized": {"model__n_estimators": [100, 200], "model__max_depth": [3, 5, 7]},
        "GradientBoosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.1]},
        "XGBoost": {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.1], "model__max_depth": [3, 5, 7]},
        "LightGBM": {"model__n_estimators": [100, 200], "model__num_leaves": [31, 63]},
        "SVM": {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]},
        "KNN": {"model__n_neighbors": [3, 5, 7], "model__weights": ["uniform", "distance"]},
        "MLP": {"model__hidden_layer_sizes": [(50,), (100,), (50, 50)], "model__alpha": [0.0001, 0.001]},
        "MLP_Regularized": {"model__hidden_layer_sizes": [(50,), (100,)], "model__alpha": [0.001, 0.01]},
        "CatBoost": {"model__depth": [4, 6], "model__iterations": [300, 500]},
        "DecisionTree": {"model__max_depth": [3, 5, None], "model__min_samples_split": [2, 5, 10]}
    }
    return param_grids

