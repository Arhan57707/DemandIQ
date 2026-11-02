
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR = REPORT_DIR / "figures"

PRIMARY_FILE = PROCESSED_DIR / "retail_demand_clean.csv"

ALTERNATIVE_FILE = (
    PROCESSED_DIR
    / "retail_demand_clean_with_duplicates.csv"
)


# ==================================================
# 2. LOAD AND VERIFY DATA
# ==================================================

def load_sales_data(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {file_path}"
        )

    print(f"\nLoading: {file_path.name}")

    df = pd.read_csv(
        file_path,
        dtype={
            "invoice": "string",
            "stock_code": "string"
        },
        parse_dates=["invoice_date"]
    )

    required_columns = [
        "invoice",
        "stock_code",
        "quantity",
        "invoice_date",
        "unit_price"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Basic safeguards
    df = df.dropna(
        subset=[
            "invoice",
            "stock_code",
            "quantity",
            "invoice_date"
        ]
    ).copy()

    # This pipeline expects positive observed sales only
    df = df[df["quantity"] > 0].copy()

    cancelled_mask = (
        df["invoice"]
        .str.upper()
        .str.startswith("C", na=False)
    )

    df = df[~cancelled_mask].copy()

    # Create a calendar date without the time component
    df["sales_date"] = (
        df["invoice_date"].dt.normalize()
    )

    print("Rows available:", f"{len(df):,}")

    return df


# ==================================================
# 3. AGGREGATE TRANSACTIONS BY PRODUCT AND DATE
# ==================================================

def aggregate_daily_sales(df):

    data = df.copy()

    # A positive price is required for ordinary
    # positive-price revenue calculation.
    positive_price = data["unit_price"] > 0

    nonpositive_price = (
        data["unit_price"].notna()
        & (data["unit_price"] <= 0)
    )

    missing_price = data["unit_price"].isna()

    data["eligible_revenue"] = (
        data["quantity"] * data["unit_price"]
    ).where(positive_price, 0.0)

    data["positive_price_units"] = (
        data["quantity"].where(positive_price, 0)
    )

    data["nonpositive_price_units"] = (
        data["quantity"].where(nonpositive_price, 0)
    )

    data["missing_price_units"] = (
        data["quantity"].where(missing_price, 0)
    )

    print("Aggregating daily product sales...")

    daily = (
        data.groupby(
            ["stock_code", "sales_date"],
            as_index=False,
            sort=True
        )
        .agg(
            daily_units_sold=(
                "quantity", "sum"
            ),

            transaction_lines=(
                "quantity", "size"
            ),

            invoice_count=(
                "invoice", "nunique"
            ),

            gross_revenue=(
                "eligible_revenue", "sum"
            ),

            units_with_positive_price=(
                "positive_price_units", "sum"
            ),

            units_with_nonpositive_price=(
                "nonpositive_price_units", "sum"
            ),

            units_with_missing_price=(
                "missing_price_units", "sum"
            )
        )
    )

    daily["daily_units_sold"] = (
        daily["daily_units_sold"].astype("int64")
    )

    daily["transaction_lines"] = (
        daily["transaction_lines"].astype("int64")
    )

    daily["invoice_count"] = (
        daily["invoice_count"].astype("int64")
    )

    return daily


# ==================================================
# 4. CREATE A PRODUCT CATALOGUE
# ==================================================

def create_product_catalog(df):

    print("\nCreating product catalogue...")

    # Find the most frequently recorded description
    # for each stock code.
    description_counts = (
        df.dropna(
            subset=["description"]
        )
        .groupby(
            ["stock_code", "description"],
            as_index=False
        )
        .size()
        .rename(
            columns={"size": "description_frequency"}
        )
        .sort_values(
            ["stock_code", "description_frequency",
             "description"],
            ascending=[True, False, True]
        )
        .drop_duplicates(
            subset=["stock_code"]
        )
        [["stock_code", "description"]]
    )

    # Create product-level statistics
    catalog_stats = (
        df.groupby("stock_code")
        .agg(
            first_sale_date=(
                "sales_date", "min"
            ),

            last_sale_date=(
                "sales_date", "max"
            ),

            total_units_sold=(
                "quantity", "sum"
            ),

            transaction_lines=(
                "quantity", "size"
            ),

            invoice_count=(
                "invoice", "nunique"
            ),

            days_with_sales=(
                "sales_date", "nunique"
            ),

            description_variants=(
                "description", "nunique"
            )
        )
        .reset_index()
    )

    catalog = catalog_stats.merge(
        description_counts,
        on="stock_code",
        how="left"
    )

    return catalog


# ==================================================
# 5. CREATE A COMPLETE DAILY PRODUCT PANEL
# ==================================================

def create_daily_panel(daily):

    print("\nCreating daily product panel...")

    first_dates = (
        daily.groupby("stock_code")["sales_date"]
        .min()
    )

    product_codes = sorted(
        daily["stock_code"].unique().tolist()
    )

    start_date = daily["sales_date"].min()
    end_date = daily["sales_date"].max()

    all_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    print("Number of products:", len(product_codes))
    print("First date:", start_date.date())
    print("Last date:", end_date.date())
    print("Calendar days:", len(all_dates))

    # Create every possible product-date combination
    full_index = pd.MultiIndex.from_product(
        [product_codes, all_dates],
        names=["stock_code", "sales_date"]
    )

    print(
        "Maximum product-date combinations:",
        f"{len(full_index):,}"
    )

    # Place actual aggregated transactions on the calendar
    panel = (
        daily.set_index(
            ["stock_code", "sales_date"]
        )
        .reindex(full_index)
    )

    # Remove dates before each product's first recorded sale.
    # We don't have evidence that a product existed before that.
    code_level = panel.index.get_level_values(
        "stock_code"
    )

    date_level = panel.index.get_level_values(
        "sales_date"
    )

    first_dates_aligned = code_level.map(
        first_dates
    )

    keep_mask = (
        date_level >= first_dates_aligned
    )

    panel = panel.loc[keep_mask].copy()

    # Days with no recorded sale become zero.
    count_columns = [
        "daily_units_sold",
        "transaction_lines",
        "invoice_count",
        "units_with_positive_price",
        "units_with_nonpositive_price",
        "units_with_missing_price"
    ]

    for column in count_columns:

        panel[column] = (
            panel[column]
            .fillna(0)
            .astype("int64")
        )

    panel["gross_revenue"] = (
        panel["gross_revenue"].fillna(0.0)
    )

    # Distinguish recorded transactions from
    # calendar dates with no recorded sale.
    panel["has_transaction"] = (
        panel["transaction_lines"] > 0
    ).astype("int8")

    panel = panel.reset_index()

    return panel


# ==================================================
# 6. CREATE TOTAL DAILY SALES
# ==================================================



def create_daily_totals(daily, primary):

    totals = (
        daily.groupby("sales_date")
        .agg(
            total_units_sold=(
                "daily_units_sold", "sum"
            ),

            products_sold=(
                "stock_code", "nunique"
            ),

            total_transaction_lines=(
                "transaction_lines", "sum"
            ),

            total_gross_revenue=(
                "gross_revenue", "sum"
            )
        )
    )

    # Calculate unique invoices directly from
    # the transaction-level data, not by adding
    # product-level invoice counts.
    invoice_totals = (
        primary.groupby("sales_date")["invoice"]
        .nunique()
        .rename("total_unique_invoices")
    )

    totals = totals.join(
        invoice_totals,
        how="left"
    )

    # Include every calendar date in the dataset period
    date_range = pd.date_range(
        start=daily["sales_date"].min(),
        end=daily["sales_date"].max(),
        freq="D"
    )

    totals = totals.reindex(
        date_range,
        fill_value=0
    )

    totals["total_unique_invoices"] = (
        totals["total_unique_invoices"]
        .fillna(0)
        .astype("int64")
    )

    totals.index.name = "sales_date"

    return totals.reset_index()



# ==================================================
# 7. COMPARE DUPLICATE-RETAINED AND CLEAN DATA
# ==================================================

def create_duplicate_comparison(
    primary_daily,
    alternative_daily
):

    print("\nComparing duplicate handling...")

    comparison = primary_daily[
        [
            "stock_code",
            "sales_date",
            "daily_units_sold"
        ]
    ].merge(
        alternative_daily[
            [
                "stock_code",
                "sales_date",
                "daily_units_sold"
            ]
        ],
        on=["stock_code", "sales_date"],
        how="outer",
        suffixes=("_primary", "_with_duplicates"),
        validate="one_to_one"
    ).fillna(0)

    extra_units = (
        comparison["daily_units_sold_with_duplicates"]
        - comparison["daily_units_sold_primary"]
    )

    extra_units_total = int(
        extra_units.sum()
    )

    primary_units = int(
        comparison["daily_units_sold_primary"].sum()
    )

    alternative_units = int(
        comparison["daily_units_sold_with_duplicates"].sum()
    )

    affected_product_days = int(
        (extra_units > 0).sum()
    )

    percentage_difference = (
        extra_units_total / primary_units * 100
        if primary_units > 0 else 0
    )

    summary = pd.DataFrame({
        "metric": [
            "primary_total_units",
            "duplicate_retained_total_units",
            "extra_units_from_duplicate_retention",
            "product_days_affected",
            "percentage_increase_vs_primary"
        ],

        "value": [
            primary_units,
            alternative_units,
            extra_units_total,
            affected_product_days,
            round(percentage_difference, 4)
        ]
    })

    return summary


# ==================================================
# 8. CREATE VISUALIZATIONS
# ==================================================

def create_visualizations(daily_totals):

    print("\nCreating visualizations...")

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Daily total units sold
    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        daily_totals["sales_date"],
        daily_totals["total_units_sold"],
        label="Daily units sold",
        alpha=0.55
    )

    ax.plot(
        daily_totals["sales_date"],
        daily_totals["total_units_sold"]
        .rolling(7, min_periods=1)
        .mean(),
        label="7-day moving average",
        linewidth=2
    )

    ax.set_title(
        "Daily Total Units Sold"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "daily_total_units_sold.png",
        dpi=150
    )

    plt.close(fig)

    # Monthly total units sold
    monthly = (
        daily_totals.set_index("sales_date")
        ["total_units_sold"]
        .resample("MS")
        .sum()
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )

    ax.set_title(
        "Monthly Total Units Sold"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Units sold")

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "monthly_total_units_sold.png",
        dpi=150
    )

    plt.close(fig)


# ==================================================
# 9. CREATE AGGREGATION REPORT
# ==================================================

def create_aggregation_report(
    primary,
    daily,
    panel,
    catalog,
    daily_totals
):

    first_date = daily["sales_date"].min()
    last_date = daily["sales_date"].max()

    zero_sales_product_days = int(
        (panel["daily_units_sold"] == 0).sum()
    )

    report = pd.DataFrame({
        "metric": [
            "clean_transaction_rows",
            "unique_products",
            "first_sales_date",
            "last_sales_date",
            "calendar_days",
            "observed_product_date_combinations",
            "dense_product_panel_rows",
            "zero_sales_product_days",
            "dates_with_any_recorded_sales",
            "dates_without_recorded_sales",
            "total_positive_units_sold"
        ],

        "value": [
            len(primary),
            len(catalog),
            str(first_date.date()),
            str(last_date.date()),
            len(daily_totals),
            len(daily),
            len(panel),
            zero_sales_product_days,
            int(
                (daily_totals["total_units_sold"] > 0).sum()
            ),
            int(
                (daily_totals["total_units_sold"] == 0).sum()
            ),
            int(daily["daily_units_sold"].sum())
        ]
    })

    return report


# ==================================================
# 10. MAIN PROGRAM
# ==================================================

def main():

    print("=" * 60)
    print("DEMANDIQ - DAILY DEMAND AGGREGATION")
    print("=" * 60)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load both Day 3 datasets
    primary = load_sales_data(
        PRIMARY_FILE
    )

    alternative = load_sales_data(
        ALTERNATIVE_FILE
    )

    # Aggregate primary cleaned data
    primary_daily = aggregate_daily_sales(
        primary
    )

    # Aggregate alternative with duplicates
    alternative_daily = aggregate_daily_sales(
        alternative
    )

    # Create product catalogue
    catalog = create_product_catalog(
        primary
    )

    # Create complete daily product panel
    panel = create_daily_panel(
        primary_daily
    )

    # Create total daily sales across all products
    daily_totals = create_daily_totals(
        primary_daily,
        primary
    )

    # Compare the duplicate handling decisions
    duplicate_comparison = create_duplicate_comparison(
        primary_daily,
        alternative_daily
    )

    # Create summary report
    aggregation_report = create_aggregation_report(
        primary,
        primary_daily,
        panel,
        catalog,
        daily_totals
    )

    # Save primary sparse daily dataset
    primary_daily.to_csv(
        PROCESSED_DIR / "daily_product_sales.csv",
        index=False
    )

    # Save alternative daily dataset
    alternative_daily.to_csv(
        PROCESSED_DIR
        / "daily_product_sales_with_duplicates.csv",
        index=False
    )

    # Save dense product-day panel
    panel.to_csv(
        PROCESSED_DIR / "daily_product_panel.csv",
        index=False
    )

    # Save product catalogue
    catalog.to_csv(
        PROCESSED_DIR / "product_catalog.csv",
        index=False
    )

    # Save total daily sales
    daily_totals.to_csv(
        PROCESSED_DIR / "daily_total_sales.csv",
        index=False
    )

    # Save reports
    duplicate_comparison.to_csv(
        REPORT_DIR / "duplicate_sensitivity_summary.csv",
        index=False
    )

    aggregation_report.to_csv(
        REPORT_DIR / "demand_aggregation_summary.csv",
        index=False
    )

    # Create charts
    create_visualizations(
        daily_totals
    )

    print("\n" + "=" * 60)
    print("DEMANDIQ AGGREGATION COMPLETED")
    print("=" * 60)

    print("\nPRIMARY DATASET")
    print("Transaction rows:", f"{len(primary):,}")
    print("Products:", f"{len(catalog):,}")

    print("\nDAILY AGGREGATION")
    print("Product-date rows:", f"{len(primary_daily):,}")

    print("\nDENSE TIME SERIES")
    print("Product-day rows:", f"{len(panel):,}")

    print(
        "Zero-sales product-days:",
        f"{(panel['daily_units_sold'] == 0).sum():,}"
    )

    print("\nDUPLICATE SENSITIVITY")
    print(duplicate_comparison.to_string(index=False))

    print("\nOUTPUTS SAVED IN:")
    print(PROCESSED_DIR)
    print(REPORT_DIR)
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
