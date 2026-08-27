"""
eda.py
------
Reusable exploratory data analysis (EDA) and visualization helpers for the
Personal Loan Portfolio Analysis project.

Libraries used: pandas, numpy, matplotlib, seaborn.

All plotting functions save their output into the `visualizations/` folder so
the figures can be committed with the repository and reused in the notebook.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so scripts run on any machine

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "personal_loan_data.csv")
VIZ_DIR = os.path.join(PROJECT_ROOT, "visualizations")

TARGET = "PersonalLoan"

EDUCATION_LABELS = {1: "Undergraduate", 2: "Graduate", 3: "Advanced/Professional"}

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"


def ensure_viz_dir(path: str = VIZ_DIR) -> str:
    """Create the visualizations directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, filename: str) -> str:
    """Save a matplotlib figure into the visualizations folder."""
    ensure_viz_dir()
    out_path = os.path.join(VIZ_DIR, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"   [saved] visualizations/{filename}")
    return out_path


# --------------------------------------------------------------------------- #
# Data loading & profiling
# --------------------------------------------------------------------------- #


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the personal loan dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected data/personal_loan_data.csv"
        )
    df = pd.read_csv(path)
    return df


def dataset_overview(df: pd.DataFrame) -> None:
    """Print shape, columns, dtypes and a preview of the dataset."""
    print("\n--- DATASET OVERVIEW ---")
    print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col:<20} {str(df[col].dtype)}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return and print a per-column missing value report."""
    missing = df.isnull().sum()
    report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_pct": (missing / len(df) * 100).round(2),
        }
    ).sort_values("missing_count", ascending=False)
    print("\n--- MISSING VALUES ---")
    if report["missing_count"].sum() == 0:
        print("No missing values found.")
    else:
        print(report[report["missing_count"] > 0].to_string())
    return report


def duplicate_report(df: pd.DataFrame, subset: Optional[Iterable[str]] = None) -> int:
    """Print and return the number of duplicated records."""
    dupes = int(df.duplicated(subset=subset).sum())
    print("\n--- DUPLICATE RECORDS ---")
    print(f"Duplicate rows: {dupes}")
    return dupes


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Print descriptive statistics for numeric columns."""
    stats = df.describe().T
    stats["range"] = stats["max"] - stats["min"]
    print("\n--- DESCRIPTIVE STATISTICS ---")
    print(stats.round(2).to_string())
    return stats


# --------------------------------------------------------------------------- #
# Feature engineering helpers
# --------------------------------------------------------------------------- #


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add a categorical AgeGroup column derived from Age."""
    bins = [0, 30, 40, 50, 60, np.inf]
    labels = ["<30", "30-39", "40-49", "50-59", "60+"]
    df = df.copy()
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    return df


def add_income_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add an income band column using quartiles."""
    df = df.copy()
    df["IncomeBand"] = pd.qcut(
        df["Income"], q=4, labels=["Low", "Lower-Mid", "Upper-Mid", "High"]
    )
    return df


def add_education_label(df: pd.DataFrame) -> pd.DataFrame:
    """Map numeric education codes to readable labels."""
    df = df.copy()
    df["EducationLabel"] = df["Education"].map(EDUCATION_LABELS)
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    return add_education_label(add_income_band(add_age_group(df)))


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #


def acceptance_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Compute loan acceptance rate grouped by a column."""
    grouped = (
        df.groupby(column, observed=True)[TARGET]
        .agg(customers="count", accepted="sum")
        .reset_index()
    )
    grouped["acceptance_rate_pct"] = (
        grouped["accepted"] / grouped["customers"] * 100
    ).round(2)
    return grouped


def correlation_with_target(df: pd.DataFrame) -> pd.Series:
    """Correlation of every numeric feature with loan acceptance."""
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr()[TARGET].drop(labels=[TARGET]).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #


def plot_target_distribution(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots()
    counts = df[TARGET].value_counts().sort_index()
    sns.barplot(x=["Declined", "Accepted"], y=counts.values, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}", ha="center", va="bottom")
    ax.set_title("Personal Loan Acceptance Distribution")
    ax.set_ylabel("Customers")
    return save_figure(fig, "01_loan_acceptance_distribution.png")


def plot_age_distribution(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="Age", bins=20, kde=True, hue=TARGET, ax=ax)
    ax.set_title("Customer Age Distribution by Loan Acceptance")
    return save_figure(fig, "02_age_distribution.png")


def plot_income_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(data=df, x="Income", bins=25, kde=True, ax=axes[0])
    axes[0].set_title("Income Distribution (all customers)")
    sns.boxplot(data=df, x=TARGET, y="Income", ax=axes[1])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Declined", "Accepted"])
    axes[1].set_title("Income vs Loan Acceptance")
    return save_figure(fig, "03_income_analysis.png")


def plot_education_analysis(df: pd.DataFrame) -> str:
    data = acceptance_rate_by(add_education_label(df), "EducationLabel")
    fig, ax = plt.subplots()
    sns.barplot(data=data, x="EducationLabel", y="acceptance_rate_pct", ax=ax)
    for i, v in enumerate(data["acceptance_rate_pct"]):
        ax.text(i, v, f"{v}%", ha="center", va="bottom")
    ax.set_title("Loan Acceptance Rate by Education Level")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_xlabel("Education")
    return save_figure(fig, "04_education_acceptance.png")


def plot_family_analysis(df: pd.DataFrame) -> str:
    data = acceptance_rate_by(df, "Family")
    fig, ax = plt.subplots()
    sns.barplot(data=data, x="Family", y="acceptance_rate_pct", ax=ax)
    ax.set_title("Loan Acceptance Rate by Family Size")
    ax.set_ylabel("Acceptance rate (%)")
    return save_figure(fig, "05_family_acceptance.png")


def plot_mortgage_analysis(df: pd.DataFrame) -> str:
    df = df.copy()
    df["HasMortgage"] = np.where(df["Mortgage"] > 0, "Mortgage", "No mortgage")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df.loc[df["Mortgage"] > 0, "Mortgage"], bins=20, ax=axes[0])
    axes[0].set_title("Mortgage Value Distribution (holders only)")
    rates = acceptance_rate_by(df, "HasMortgage")
    sns.barplot(data=rates, x="HasMortgage", y="acceptance_rate_pct", ax=axes[1])
    axes[1].set_title("Loan Acceptance by Mortgage Ownership")
    axes[1].set_ylabel("Acceptance rate (%)")
    return save_figure(fig, "06_mortgage_analysis.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["ID"], errors="ignore")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap of Customer Attributes")
    return save_figure(fig, "07_correlation_heatmap.png")


def plot_income_vs_ccavg(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Income", y="CCAvg", hue=TARGET, alpha=0.7, ax=ax)
    ax.set_title("Income vs Average Credit Card Spend")
    ax.set_ylabel("Avg monthly card spend ($000s)")
    return save_figure(fig, "08_income_vs_ccavg.png")


def plot_age_group_acceptance(df: pd.DataFrame) -> str:
    data = acceptance_rate_by(add_age_group(df), "AgeGroup")
    fig, ax = plt.subplots()
    sns.barplot(data=data, x="AgeGroup", y="acceptance_rate_pct", ax=ax)
    ax.set_title("Loan Acceptance Rate by Age Group")
    ax.set_ylabel("Acceptance rate (%)")
    return save_figure(fig, "09_age_group_acceptance.png")


def generate_all_visualizations(df: pd.DataFrame) -> List[str]:
    """Render every chart used in the report."""
    print("\n--- GENERATING VISUALIZATIONS ---")
    return [
        plot_target_distribution(df),
        plot_age_distribution(df),
        plot_income_distribution(df),
        plot_education_analysis(df),
        plot_family_analysis(df),
        plot_mortgage_analysis(df),
        plot_correlation_heatmap(df),
        plot_income_vs_ccavg(df),
        plot_age_group_acceptance(df),
    ]


if __name__ == "__main__":
    frame = load_data()
    dataset_overview(frame)
    missing_value_report(frame)
    duplicate_report(frame)
