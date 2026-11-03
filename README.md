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

### Day 4 - Daily Demand Aggregation and Time-Series Preparation

- Aggregated transaction-level data by product and date.
- Calculated daily units sold for each stock code.
- Calculated transaction-line and invoice counts.
- Calculated positive-price gross revenue.
- Separated units associated with nonpositive or missing prices.
- Created a product catalogue.
- Generated a sparse daily product-sales dataset.
- Generated a dense daily product panel with zero-sales days.
- Retained a separate daily aggregation containing duplicates.
- Compared the impact of duplicate handling on measured demand.
- Generated total daily sales across all products.
- Created daily and monthly sales visualizations.
- Implemented aggregation reconciliation and validation checks.

### Day 4 Output Files

- `daily_product_sales.csv`
- `daily_product_sales_with_duplicates.csv`
- `daily_product_panel.csv`
- `product_catalog.csv`
- `daily_total_sales.csv`
- `demand_aggregation_summary.csv`
- `duplicate_sensitivity_summary.csv`

### Time-Series Definition

The forecasting target is observed positive units sold,
aggregated at the product-day level.

Dates with no recorded transaction are represented as
zero in the dense product panel.

Products are included from their first recorded sale
through the final date in the dataset.

Zero recorded sales do not necessarily establish zero
underlying demand because stock availability and
stockout information are not available.

Duplicate removal is treated as a documented analytical
decision and compared against an alternative dataset
that retains exact duplicate transaction records.

## Feature Engineering

DemandIQ transforms the dense daily product-demand panel into
a machine-learning feature dataset.

### Dataset Overview

- Product-day records: 3,084,067
- Unique products: 4,984
- Date range: December 1, 2009 – December 9, 2011
- Target variable: daily_units_sold
- Engineered columns: 37

### Feature Groups

**Calendar Features**
- Day of week, day of month, day of year
- Week of year, month, quarter, year
- Weekend, Saturday, Sunday and December indicators
- Cyclical weekday and monthly encodings

**Historical Demand Features**
- Lag features: 1, 7, 14 and 28 days
- Rolling means: 7, 14 and 28 days
- Rolling standard deviations: 7, 14 and 28 days

**Product Metadata**
- Product stock code
- Product description
- Description missingness indicator

### Data Leakage Prevention

Historical demand features are constructed using shifted
demand values so that the current prediction date's actual
demand is excluded from lag and rolling calculations.

Full-period product sales statistics are excluded from
predictor metadata.

Same-day transaction outcomes and the target variable
must be excluded from model inputs.

### Missing Data Handling

- 64 products had missing descriptions.
- 44,763 product-day records were affected.
- Missing descriptions were represented as "Unknown".
- A description_missing indicator preserves the original
  missingness information.
- Historical lag and rolling warm-up NaNs are retained.

### Validation

The independent feature validation script completed
31 checks with 31 passing and 0 failing.

Validation includes:
- Row and product counts
- Demand reconciliation
- Duplicate product-date checks
- Product metadata integrity
- Calendar feature consistency
- Lag and rolling missing-value counts
- Non-negative demand features
- Cyclical calendar feature checks
