# Personal Loan Portfolio Analysis

**A Python-based Data Science internship project** analysing a retail bank's personal
loan portfolio using **Pandas, NumPy, Matplotlib and Seaborn**.

The goal is to understand which customer characteristics — age, income, education,
family size, mortgage ownership and banking relationships — drive personal loan
acceptance, and to turn those findings into actionable business insights.

## Tech stack

- **Language:** Python 3
- **Data manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Notebook environment:** jupyter / notebook

## Project structure

```
personal-loan-portfolio-analysis/
├── data/
│   └── personal_loan_data.csv          # 800-record personal loan dataset
├── src/
│   ├── loan_analysis.py                # Main end-to-end analysis pipeline
│   └── eda.py                          # Reusable EDA + visualization functions
├── notebooks/
│   └── Personal_Loan_Portfolio_Analysis.ipynb
├── visualizations/                     # Generated charts (PNG)
├── main.py                             # Convenience entry point
├── pyproject.toml                      # Python project metadata
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

`data/personal_loan_data.csv` contains 800 customer records with the following columns:

| Column | Description |
| --- | --- |
| `ID` | Customer identifier |
| `Age` | Customer age in years |
| `Experience` | Years of professional experience |
| `Income` | Annual income (in $000s) |
| `ZIPCode` | Residential ZIP code |
| `Family` | Family size |
| `CCAvg` | Average monthly credit card spend ($000s) |
| `Education` | 1 = Undergraduate, 2 = Graduate, 3 = Advanced/Professional |
| `Mortgage` | Value of house mortgage ($000s), 0 if none |
| `PersonalLoan` | Target: 1 if the customer accepted a personal loan |
| `SecuritiesAccount` | Holds a securities account |
| `CDAccount` | Holds a certificate of deposit |
| `Online` | Uses online banking |
| `CreditCard` | Holds a bank credit card |

## Getting started

```sh
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full analysis
python src/loan_analysis.py

# 3. Or explore interactively
jupyter notebook notebooks/Personal_Loan_Portfolio_Analysis.ipynb
```

Charts are written to the `visualizations/` folder.

## What the analysis does

1. Loads `data/personal_loan_data.csv`
2. Inspects dataset shape, columns and data types
3. Reports missing values per column
4. Detects duplicate records
5. Cleans the data (drops duplicates, fixes negative experience, imputes missing values)
6. Produces descriptive statistics
7. Analyses customer **age** and age groups
8. Analyses **income** and income bands
9. Analyses **education** levels
10. Analyses **family size**
11. Analyses **mortgage** ownership and value
12. Analyses overall **personal loan acceptance**, including product-holding segments
13. Measures relationships between customer characteristics and loan acceptance (correlations, group profiles)
14. Generates nine charts with matplotlib/seaborn
15. Prints data-driven business insights

## Key findings (from the current dataset)

- Roughly **6.8%** of customers accept a personal loan, so targeting precision matters more than campaign reach.
- **Income is the strongest driver** — the highest income quartile converts at about **22.8%**, more than three times the portfolio average.
- **Average credit card spend** is the second strongest signal (correlation ≈ 0.39 with acceptance).
- Acceptors earn on average **~$130k** versus **~$58k** for non-acceptors.
- **Age is a weak standalone predictor**; education and family size act as useful secondary segmentation variables.

## Visualizations produced

`01` acceptance distribution · `02` age distribution · `03` income analysis ·
`04` education acceptance · `05` family size acceptance · `06` mortgage analysis ·
`07` correlation heatmap · `08` income vs card spend · `09` acceptance by age group
