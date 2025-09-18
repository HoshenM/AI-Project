import os
import pandas as pd
import kagglehub

def load_data():
    """Download and load the Stroke dataset from Kaggle."""
    print("\nLoading Dataset...")
    path = kagglehub.dataset_download(
        "shriyashjagtap/stroke-diagnosis-and-health-metrics-data"
    )
    df = pd.read_csv(os.path.join(path, "stroke_data.csv"))

    print(f"Dataset Shape: {df.shape}")
    print("Dataset Source: https://www.kaggle.com/shriyashjagtap/stroke-diagnosis-and-health-metrics-data")
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset Description:")
    print(df.describe())

    return df
