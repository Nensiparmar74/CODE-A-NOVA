"""
CODE-A-NOVA Month 3 — Level 1(b): Customer Segmentation

Dataset:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Download Mall_Customers.csv and put it in the data/ folder.
Run: python customer_segmentation.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Mall_Customers.csv"
OUTPUT_DIR = ROOT / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Download Mall_Customers.csv from the Kaggle link in README.md "
            "and save it in the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    required = [
        "CustomerID", "Gender", "Age",
        "Annual Income (k$)", "Spending Score (1-100)"
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "The CSV does not have the expected columns: " + ", ".join(missing)
        )

    print("\n--- First five rows ---")
    print(df.head())
    print("\n--- Dataset information ---")
    df.info()
    print("\n--- Missing values ---")
    print(df.isnull().sum())
    print("\n--- Duplicate rows ---")
    print(df.duplicated().sum())
    print("\n--- Summary statistics ---")
    print(df.describe(include="all"))

    df = df.drop_duplicates().copy()
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = df[features].copy()
    X = X.fillna(X.median(numeric_only=True))

    sns.set_theme(style="whitegrid")

    # Explore age distribution.
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"].dropna(), kde=True)
    plt.title("Customer Age Distribution")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_distribution.png", dpi=150)
    plt.close()

    # Explore income and spending.
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=features[0], y=features[1])
    plt.title("Annual Income vs Spending Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "income_spending.png", dpi=150)
    plt.close()

    # Standardize features before distance-based clustering.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method to compare candidate cluster counts.
    k_values = list(range(1, 11))
    inertias = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow Method for Selecting K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_method.png", dpi=150)
    plt.close()

    # Five clusters is a candidate commonly explored for this dataset.
    # Review the elbow plot and justify or adjust this choice in your report.
    chosen_k = 5
    model = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(X_scaled)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x=features[0],
        y=features[1],
        hue="Cluster",
        palette="tab10",
        s=85
    )
    plt.title(f"Customer Segments Using K-Means (K={chosen_k})")
    plt.xlabel("Annual Income (thousands of dollars)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "customer_segments.png", dpi=150)
    plt.close()

    profile = df.groupby("Cluster").agg(
        Customer_Count=("CustomerID", "count"),
        Average_Age=("Age", "mean"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending=("Spending Score (1-100)", "mean")
    ).round(2)

    print("\n--- Customer Segment Profiles ---")
    print(profile)

    df.to_csv(OUTPUT_DIR / "segmented_customers.csv", index=False)
    profile.to_csv(OUTPUT_DIR / "segment_profiles.csv")
    print(f"\nAnalysis complete. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
