"""
loan_analysis.py
----------------
Personal Loan Portfolio Analysis — main analysis pipeline.

This script performs an end-to-end analysis of a retail bank's personal loan
portfolio using pandas, numpy, matplotlib and seaborn:

    1.  Load data/personal_loan_data.csv
    2.  Inspect shape, columns and data types
    3.  Check missing values
    4.  Check duplicate records
    5.  Clean the data
    6.  Descriptive statistics
    7.  Age analysis
    8.  Income analysis
    9.  Education analysis
    10. Family size analysis
    11. Mortgage analysis
    12. Personal loan acceptance analysis
    13. Relationships between customer characteristics and loan acceptance
    14. Charts (saved to visualizations/)
    15. Data-driven business insights

Run with:  python src/loan_analysis.py
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from eda import (
    DATA_PATH,
    EDUCATION_LABELS,
    TARGET,
    acceptance_rate_by,
    add_age_group,
    add_education_label,
    add_income_band,
    correlation_with_target,
    descriptive_statistics,
    dataset_overview,
    duplicate_report,
    enrich,
    generate_all_visualizations,
    load_data,
    missing_value_report,
)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title.upper())
    print("=" * 78)


# --------------------------------------------------------------------------- #
# 1-5. Load, inspect and clean
# --------------------------------------------------------------------------- #


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset: duplicates, missing values, invalid records."""
    header("step 5: data cleaning")
    rows_before = len(df)

    # Drop exact duplicate records
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {rows_before - len(df)} duplicate rows.")

    # Negative work experience is a known data-entry issue -> use absolute value
    if "Experience" in df.columns:
        negatives = int((df["Experience"] < 0).sum())
        df["Experience"] = df["Experience"].abs()
        print(f"Corrected {negatives} negative Experience values.")

    # Impute missing numeric values
    for col in ["Income", "CCAvg", "Mortgage", "Age", "Experience"]:
        if col in df.columns and df[col].isnull().any():
            median = df[col].median()
            n_missing = int(df[col].isnull().sum())
            df[col] = df[col].fillna(median)
            print(f"Filled {n_missing} missing '{col}' values with median {median:.2f}.")

    # Impute missing categorical/count values with the mode
    for col in ["Family", "Education"]:
        if col in df.columns and df[col].isnull().any():
            mode = df[col].mode().iloc[0]
            n_missing = int(df[col].isnull().sum())
            df[col] = df[col].fillna(mode)
            print(f"Filled {n_missing} missing '{col}' values with mode {mode}.")

    # Cast count-like columns back to integers
    for col in ["Family", "Education", "Age", "Experience", TARGET]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    print(f"Clean dataset shape: {df.shape}")
    return df


# --------------------------------------------------------------------------- #
# 7-11. Attribute analyses
# --------------------------------------------------------------------------- #


def analyze_age(df: pd.DataFrame) -> pd.DataFrame:
    header("step 7: customer age analysis")
    print(
        f"Mean age: {df['Age'].mean():.1f} | Median: {df['Age'].median():.0f} | "
        f"Min: {df['Age'].min()} | Max: {df['Age'].max()} | Std: {df['Age'].std():.1f}"
    )
    by_group = acceptance_rate_by(add_age_group(df), "AgeGroup")
    print("\nAcceptance rate by age group:")
    print(by_group.to_string(index=False))
    return by_group


def analyze_income(df: pd.DataFrame) -> pd.DataFrame:
    header("step 8: income analysis")
    print(
        f"Mean income: ${df['Income'].mean():.1f}k | Median: ${df['Income'].median():.1f}k | "
        f"90th pct: ${df['Income'].quantile(0.9):.1f}k"
    )
    accepted = df.loc[df[TARGET] == 1, "Income"].mean()
    declined = df.loc[df[TARGET] == 0, "Income"].mean()
    print(f"Average income of acceptors: ${accepted:.1f}k vs non-acceptors: ${declined:.1f}k")
    by_band = acceptance_rate_by(add_income_band(df), "IncomeBand")
    print("\nAcceptance rate by income band:")
    print(by_band.to_string(index=False))
    return by_band


def analyze_education(df: pd.DataFrame) -> pd.DataFrame:
    header("step 9: education analysis")
    labelled = add_education_label(df)
    print("Customer counts by education level:")
    print(labelled["EducationLabel"].value_counts().to_string())
    by_edu = acceptance_rate_by(labelled, "EducationLabel")
    print("\nAcceptance rate by education level:")
    print(by_edu.to_string(index=False))
    return by_edu


def analyze_family(df: pd.DataFrame) -> pd.DataFrame:
    header("step 10: family size analysis")
    print(f"Average family size: {df['Family'].mean():.2f}")
    by_family = acceptance_rate_by(df, "Family")
    print("\nAcceptance rate by family size:")
    print(by_family.to_string(index=False))
    return by_family


def analyze_mortgage(df: pd.DataFrame) -> pd.DataFrame:
    header("step 11: mortgage analysis")
    holders = df[df["Mortgage"] > 0]
    share = len(holders) / len(df) * 100
    print(f"Customers with a mortgage: {len(holders)} ({share:.1f}%)")
    if len(holders):
        print(
            f"Average mortgage (holders): ${holders['Mortgage'].mean():.1f}k | "
            f"Max: ${holders['Mortgage'].max():.0f}k"
        )
    tmp = df.copy()
    tmp["HasMortgage"] = np.where(tmp["Mortgage"] > 0, "Mortgage", "No mortgage")
    by_mortgage = acceptance_rate_by(tmp, "HasMortgage")
    print("\nAcceptance rate by mortgage ownership:")
    print(by_mortgage.to_string(index=False))
    return by_mortgage


def analyze_loan_acceptance(df: pd.DataFrame) -> Dict[str, float]:
    header("step 12: personal loan acceptance analysis")
    total = len(df)
    accepted = int(df[TARGET].sum())
    rate = accepted / total * 100
    print(f"Total customers      : {total}")
    print(f"Loans accepted       : {accepted}")
    print(f"Overall acceptance   : {rate:.2f}%")

    for flag, label in [
        ("SecuritiesAccount", "securities account"),
        ("CDAccount", "certificate of deposit"),
        ("Online", "online banking"),
        ("CreditCard", "bank credit card"),
    ]:
        if flag in df.columns:
            rates = acceptance_rate_by(df, flag)
            with_flag = rates.loc[rates[flag] == 1, "acceptance_rate_pct"]
            without = rates.loc[rates[flag] == 0, "acceptance_rate_pct"]
            if len(with_flag) and len(without):
                print(
                    f"  With {label:<24}: {with_flag.iloc[0]:5.2f}%  |  "
                    f"without: {without.iloc[0]:5.2f}%"
                )

    return {"total": total, "accepted": accepted, "acceptance_rate": rate}


def analyze_relationships(df: pd.DataFrame) -> pd.Series:
    header("step 13: characteristics vs loan acceptance")
    corr = correlation_with_target(df.drop(columns=["ID"], errors="ignore"))
    print("Correlation of each attribute with PersonalLoan:")
    print(corr.round(3).to_string())

    print("\nMean attribute values by loan outcome:")
    profile = df.groupby(TARGET)[["Age", "Income", "CCAvg", "Mortgage", "Family", "Education"]].mean()
    profile.index = ["Declined", "Accepted"]
    print(profile.round(2).to_string())
    return corr


# --------------------------------------------------------------------------- #
# 15. Business insights
# --------------------------------------------------------------------------- #


def business_insights(df: pd.DataFrame, corr: pd.Series) -> List[str]:
    header("step 15: data-driven business insights")
    insights: List[str] = []

    rate = df[TARGET].mean() * 100
    insights.append(
        f"Only {rate:.2f}% of the portfolio holds a personal loan, so the campaign "
        f"base rate is low and targeting precision matters more than reach."
    )

    inc = add_income_band(df)
    top_band = acceptance_rate_by(inc, "IncomeBand").sort_values(
        "acceptance_rate_pct", ascending=False
    ).iloc[0]
    insights.append(
        f"Income is the strongest driver: the '{top_band['IncomeBand']}' income band "
        f"converts at {top_band['acceptance_rate_pct']:.2f}%, well above the "
        f"{rate:.2f}% portfolio average."
    )

    edu = acceptance_rate_by(add_education_label(df), "EducationLabel").sort_values(
        "acceptance_rate_pct", ascending=False
    ).iloc[0]
    insights.append(
        f"{edu['EducationLabel']} customers accept at {edu['acceptance_rate_pct']:.2f}%, "
        f"making education a useful secondary segmentation variable."
    )

    fam = acceptance_rate_by(df, "Family").sort_values(
        "acceptance_rate_pct", ascending=False
    ).iloc[0]
    insights.append(
        f"Households of {int(fam['Family'])} show the highest acceptance "
        f"({fam['acceptance_rate_pct']:.2f}%), suggesting family-related borrowing needs."
    )

    mort = df[df["Mortgage"] > 0][TARGET].mean() * 100
    nomort = df[df["Mortgage"] == 0][TARGET].mean() * 100
    direction = "higher" if mort > nomort else "lower"
    insights.append(
        f"Mortgage holders convert at {mort:.2f}% versus {nomort:.2f}% for non-holders — "
        f"a {direction} propensity that should be reflected in credit-risk scoring."
    )

    top_two = corr.head(2)
    insights.append(
        "Strongest positive correlations with acceptance: "
        + ", ".join(f"{k} ({v:.2f})" for k, v in top_two.items())
        + ". These belong in any propensity model built next."
    )

    age_top = acceptance_rate_by(add_age_group(df), "AgeGroup").sort_values(
        "acceptance_rate_pct", ascending=False
    ).iloc[0]
    insights.append(
        f"Age is a weak standalone predictor (corr {corr.get('Age', float('nan')):.2f}); "
        f"the best-performing bracket is {age_top['AgeGroup']} at "
        f"{age_top['acceptance_rate_pct']:.2f}%."
    )

    for i, text in enumerate(insights, start=1):
        print(f"\n{i}. {text}")
    return insights


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run_analysis(path: str = DATA_PATH) -> pd.DataFrame:
    header("personal loan portfolio analysis")
    print(f"Dataset: {path}")
    print(f"Education encoding: {EDUCATION_LABELS}")

    header("steps 2-4: shape, columns, missing values, duplicates")
    raw = load_data(path)
    dataset_overview(raw)
    missing_value_report(raw)
    duplicate_report(raw)

    df = clean_data(raw)

    header("step 6: descriptive statistics")
    descriptive_statistics(df.drop(columns=["ID", "ZIPCode"], errors="ignore"))

    analyze_age(df)
    analyze_income(df)
    analyze_education(df)
    analyze_family(df)
    analyze_mortgage(df)
    analyze_loan_acceptance(df)
    corr = analyze_relationships(df)

    header("step 14: charts")
    generate_all_visualizations(df)

    business_insights(df, corr)

    header("analysis complete")
    enriched = enrich(df)
    print(f"Final analysed dataset: {enriched.shape[0]} rows x {enriched.shape[1]} columns")
    return enriched


if __name__ == "__main__":
    run_analysis()
