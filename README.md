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
