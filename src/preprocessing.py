# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df, num_features, cat_features, target_col="Stroke"):
    """
    Preprocess the stroke dataset:
    - Drop missing values
    - Split into features/target
    - Build ColumnTransformer (scaling + one-hot encoding)
    - Split into train/validation/test sets (70/15/15)
    
    Returns:
        X_train, X_valid, X_test, y_train, y_valid, y_test, preprocessor
    """

    print("\nPreprocessing Data...")

    # Drop missing values
    df = df.dropna()
    print(f"Dataset shape after removing missing values: {df.shape}")

    # Features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features)
        ],
        remainder="passthrough"
    )

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_valid.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training set class distribution: {y_train.value_counts().to_dict()}")

    return X_train, X_valid, X_test, y_train, y_valid, y_test, preprocessor
