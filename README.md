# DemandIQ

## Smart Demand Forecasting and Inventory Intelligence

### Project Overview

DemandIQ is an end-to-end Data Science project
designed to analyse historical retail sales,
forecast future product demand and support
inventory planning decisions.

### Main Objectives

- Analyse historical retail sales.
- Identify product-level demand patterns.
- Build and evaluate forecasting models.
- Estimate future product demand.
- Identify potential inventory risks.
- Generate inventory recommendations.
- Present results through an interactive dashboard.

### Technology Stack

- Python
- pandas
- NumPy
- SQL
- scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Git and GitHub

### Development Status

Day 1 - Environment and project setup.


## Development Progress

### Day 1 - Project Setup
- Created the DemandIQ project structure.
- Configured Python virtual environment.
- Installed the required libraries.
- Connected the local project to GitHub.

### Day 2 - Data Understanding and Exploration
- Inspected the Online Retail II Excel workbook.
- Examined individual worksheets and data types.
- Combined the available worksheets.
- Standardised column names.
- Created transaction-level data quality reports.
- Investigated missing values, duplicates, cancellations,
  negative quantities, and invalid prices.
- Generated initial exploratory visualizations.

## Dataset

DemandIQ uses the Online Retail II transaction dataset.

The original Excel workbook is stored locally in `data/raw/`.
The combined transaction dataset is stored in `data/processed/`.

Raw and processed datasets are excluded from Git version control.

## Initial Analysis

The initial exploratory analysis includes:

1. Monthly gross sales.
2. Top countries by invoice count.
3. Top products by units sold.

These visualizations are exploratory. Data cleaning, return
handling, and final demand definitions will be documented
before the forecasting stage.

### Day 3 - Data Cleaning and Preparation

- Created a reusable data-cleaning pipeline.
- Standardised invoice IDs, stock codes, dates,
  quantities, and prices.
- Identified exact duplicate transaction records.
- Created a duplicate review file.
- Created a primary dataset with exact duplicate
  copies removed.
- Created an alternative dataset retaining duplicates.
- Excluded cancelled invoice identifiers from
  positive-demand data.
- Excluded zero and negative quantities from the
  primary positive-demand dataset.
- Preserved missing customer IDs and descriptions.
- Retained nonpositive prices with a revenue eligibility flag.
- Separated negative-quantity transactions for
  future return analysis.
- Created a data-cleaning summary report.
- Implemented validation checks for the cleaned dataset.

### Data Cleaning Decisions

The primary demand dataset uses positive quantities,
valid invoice identifiers, valid stock codes, and
valid invoice dates.

Exact duplicate business records are reduced to one
record in the primary dataset. An alternative version
retaining duplicates is maintained for sensitivity
analysis.

Customer IDs and descriptions are not mandatory for
product-demand analysis and are therefore retained
when missing.

Nonpositive prices are flagged rather than automatically
removed because unit demand and revenue eligibility
are different concepts.

The original dataset remains unchanged.

### Day 3 Output Files

- `retail_demand_clean.csv`
- `retail_demand_clean_with_duplicates.csv`
- `retail_returns_review.csv`
- `duplicate_records_review.csv`
- `data_cleaning_summary.csv`
