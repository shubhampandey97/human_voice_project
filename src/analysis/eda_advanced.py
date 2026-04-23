import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "vocal_gender_features_new.csv"
REPORTS_PATH = BASE_DIR / "reports" / "eda"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

def run_eda():
    print("Running Advanced EDA...")

    df = pd.read_csv(DATA_PATH)

    print("Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())

    # -----------------------------
    # Target Distribution
    # -----------------------------
    plt.figure()
    sns.countplot(x='label', data=df)
    plt.title("Target Distribution")
    plt.savefig(REPORTS_PATH / "target_distribution.png")
    plt.close()

    # -----------------------------
    # Correlation with target
    # -----------------------------
    corr = df.corr(numeric_only=True)['label'].sort_values(ascending=False)

    corr.to_csv(REPORTS_PATH / "correlation_with_target.csv")

    # -----------------------------
    # Mutual Information
    # -----------------------------
    X = df.drop("label", axis=1)
    y = df["label"]

    mi = mutual_info_classif(X, y)
    mi_df = pd.DataFrame({"Feature": X.columns, "MI Score": mi})
    mi_df = mi_df.sort_values(by="MI Score", ascending=False)

    mi_df.to_csv(REPORTS_PATH / "mutual_info.csv", index=False)

    # -----------------------------
    # Random Forest Importance
    # -----------------------------
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    importances.to_csv(REPORTS_PATH / "feature_importance.csv")

    plt.figure(figsize=(10,6))
    importances.head(15).plot(kind='barh')
    plt.title("Top 15 Features")
    plt.gca().invert_yaxis()
    plt.savefig(REPORTS_PATH / "feature_importance.png")
    plt.close()

    # -----------------------------
    # Skewness
    # -----------------------------
    skewness = X.apply(skew)
    skewness.to_csv(REPORTS_PATH / "skewness.csv")

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------
    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(REPORTS_PATH / "correlation_heatmap.png")
    plt.close()

    # -----------------------------
    # High Correlation Features
    # -----------------------------
    corr_matrix = df.corr(numeric_only=True).abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr = [col for col in upper.columns if any(upper[col] > 0.9)]

    with open(REPORTS_PATH / "high_correlation.txt", "w") as f:
        for col in high_corr:
            f.write(col + "\n")

    print("✅ EDA completed. Reports saved in /reports/eda")

if __name__ == "__main__":
    run_eda()