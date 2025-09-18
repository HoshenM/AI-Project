# src/eda_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def run_eda(df, pdf_path="figures/eda_report.pdf"):
    """
    Perform exploratory data analysis (EDA) on the dataset.
    Saves plots into a PDF and returns lists of numerical and categorical features.
    """
    print("\nPerforming Enhanced Exploratory Data Analysis...")

    with PdfPages(pdf_path) as pp:
        # Missing values
        print("\nMissing values per column:")
        missing_vals = df.isnull().sum()
        print(missing_vals[missing_vals > 0])

        # Duplicates
        duplicates = df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")

        # Target distribution
        print("\nTarget variable distribution:")
        target_dist = df['Stroke'].value_counts()
        print(target_dist)
        print(f"Class imbalance ratio (No Stroke/Stroke): {target_dist[0] / target_dist[1]:.2f}")

        # Feature types
        num_features = df.select_dtypes(include=np.number).columns.tolist()
        if 'Stroke' in num_features:
            num_features.remove('Stroke')
        cat_features = df.select_dtypes(exclude=np.number).columns.tolist()

        print(f"\nNumeric features ({len(num_features)}): {num_features}")
        print(f"Categorical features ({len(cat_features)}): {cat_features}")

        # Categorical features analysis
        for col in cat_features:
            print(f"\n{col} value counts:")
            print(df[col].value_counts())

        # === Visualizations ===
        # Class balance
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="Stroke")
        plt.title("Class Balance (Stroke vs No Stroke)", fontsize=14, fontweight='bold')
        plt.xlabel("Stroke (0=No, 1=Yes)")
        plt.ylabel("Count")
        for i, v in enumerate(df['Stroke'].value_counts()):
            plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
        pp.savefig(bbox_inches='tight'); plt.close()

        # Feature distributions
        df.hist(figsize=(15, 12), bins=25, edgecolor='black', alpha=0.7)
        plt.suptitle("Feature Distributions", fontsize=16, fontweight='bold')
        plt.tight_layout()
        pp.savefig(bbox_inches='tight'); plt.close()

        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap="RdBu_r", center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
        pp.savefig(bbox_inches='tight'); plt.close()

        # Feature correlation with target
        if len(num_features) > 0:
            correlations = df[num_features + ['Stroke']].corr()['Stroke'].abs().sort_values(ascending=False)[1:]
            plt.figure(figsize=(10, 6))
            correlations.plot(kind='bar')
            plt.title("Feature Correlations with Target", fontsize=14, fontweight='bold')
            plt.xlabel("Features")
            plt.ylabel("Absolute Correlation")
            plt.xticks(rotation=45)
            pp.savefig(bbox_inches='tight'); plt.close()
            print("\nFeature correlations with target:")
            print(correlations)

        # Outliers detection
        df_num = df.select_dtypes(include=np.number)
        z_scores = np.abs((df_num - df_num.mean()) / df_num.std())
        outliers_count = (z_scores > 3).sum().sum()
        print(f"\nOutliers detected (Z-score > 3): {outliers_count}")

        # Box plots for numeric features by target
        if len(num_features) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            for i, col in enumerate(num_features[:4]):
                sns.boxplot(data=df, x='Stroke', y=col, ax=axes[i])
                axes[i].set_title(f'{col} by Stroke Status')
            plt.tight_layout()
            pp.savefig(bbox_inches='tight'); plt.close()

    return num_features, cat_features
