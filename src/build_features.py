
from pathlib import Path
import pandas as pd
import numpy as np


# ==================================================
# DEMANDIQ - DAY 5: FEATURE ENGINEERING
# ==================================================

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "daily_product_panel.csv"
)
PRODUCT_CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "product_catalog.csv"
)
FEATURE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "daily_product_features.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "feature_engineering_summary.csv"
)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_panel(path):

    print("\nLoading daily product panel...")

    df = pd.read_csv(
        path,
        dtype={"stock_code": "string"},
        parse_dates=["sales_date"],
    )

    print("Dataset loaded successfully.")

    return df


# --------------------------------------------------
# 3. INPUT DATA VALIDATION
# --------------------------------------------------

def validate_input(df):

    print("\nChecking input dataset...")

    required_columns = [
        "stock_code",
        "sales_date",
        "daily_units_sold",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["stock_code"].isna().any():
        raise ValueError("Missing product identifiers found.")

    if df["sales_date"].isna().any():
        raise ValueError("Invalid or missing dates found.")

    if (df["daily_units_sold"] < 0).any():
        raise ValueError("Negative demand values found.")

    duplicated_rows = df.duplicated(
        subset=["stock_code", "sales_date"]
    ).sum()

    if duplicated_rows > 0:
        raise ValueError(
            f"Found {duplicated_rows} duplicate product-date rows."
        )

    print("Input validation passed.")

    print(f"Rows: {len(df):,}")
    print(f"Products: {df['stock_code'].nunique():,}")

    print(
        "Date range:",
        df["sales_date"].min(),
        "to",
        df["sales_date"].max(),
    )


# --------------------------------------------------
# 4. CALENDAR FEATURES
# --------------------------------------------------

def add_calendar_features(df):

    print("\nCreating calendar features...")

    df["day_of_week"] = (
        df["sales_date"].dt.day_of_week
    )

    df["day_of_month"] = (
        df["sales_date"].dt.day
    )

    df["day_of_year"] = (
        df["sales_date"].dt.day_of_year
    )

    df["week_of_year"] = (
        df["sales_date"]
        .dt.isocalendar()
        .week
        .astype("int16")
    )

    df["month"] = (
        df["sales_date"].dt.month
    )

    df["quarter"] = (
        df["sales_date"].dt.quarter
    )

    df["year"] = (
        df["sales_date"].dt.year
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype("int8")

    df["is_saturday"] = (
        df["day_of_week"] == 5
    ).astype("int8")

    df["is_sunday"] = (
        df["day_of_week"] == 6
    ).astype("int8")

    df["is_december"] = (
        df["month"] == 12
    ).astype("int8")

    # Cyclical weekday encoding

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # Cyclical month encoding

    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )

    print("Calendar features created.")

    return df


# --------------------------------------------------
# 5. LAG FEATURES
# --------------------------------------------------

def add_lag_features(df):

    print("\nCreating historical demand lag features...")

    # Group demand history separately for each product.
    grouped_demand = df.groupby(
        "stock_code",
        sort=False
    )["daily_units_sold"]

    # Create demand lags.
    lag_periods = [1, 7, 14, 28]

    for lag in lag_periods:

        feature_name = f"lag_{lag}"

        df[feature_name] = grouped_demand.shift(lag)

        print(f"Created: {feature_name}")

    print("All lag features created successfully.")

    return df
    
# --------------------------------------------------
# 6. ROLLING DEMAND FEATURES
# --------------------------------------------------

def add_rolling_features(df):

    print("\nCreating rolling demand features...")

    windows = [7, 14, 28]

    for window in windows:

        print(
            f"Creating rolling features for {window} days..."
        )

        # Calculate the rolling mean using historical demand only.
        df[f"rolling_mean_{window}"] = (
            df.groupby(
                "stock_code",
                sort=False
            )["daily_units_sold"]
            .transform(
                lambda series: (
                    series.shift(1)
                    .rolling(
                        window=window,
                        min_periods=window
                    )
                    .mean()
                )
            )
        )

    print("All rolling means created successfully.")

    return df

# --------------------------------------------------
# 7. ROLLING DEMAND VARIABILITY
# --------------------------------------------------

def add_rolling_volatility_features(df):

    print("\nCreating rolling demand variability features...")

    windows = [7, 14, 28]

    for window in windows:

        print(
            f"Creating rolling standard deviation for "
            f"{window} days..."
        )

        df[f"rolling_std_{window}"] = (
            df.groupby(
                "stock_code",
                sort=False
            )["daily_units_sold"]
            .transform(
                lambda series: (
                    series.shift(1)
                    .rolling(
                        window=window,
                        min_periods=window
                    )
                    .std()
                )
            )
        )

    print(
        "Rolling standard deviation features created."
    )

    return df


# --------------------------------------------------
# PRODUCT FEATURES
# --------------------------------------------------

def add_product_features(df):

    print("\nAdding product information...")

    # Load only the product identifier and description.
    # Lifetime sales statistics are deliberately excluded.
    catalog = pd.read_csv(
        PRODUCT_CATALOG_PATH,
        dtype={"stock_code": "string"},
        usecols=["stock_code", "description"],
    )

    # Remove accidental whitespace from identifiers.
    catalog["stock_code"] = (
        catalog["stock_code"].str.strip()
    )

    # Check that catalogue identifiers are unique.
    duplicate_products = catalog[
        catalog["stock_code"].duplicated(keep=False)
    ]

    if not duplicate_products.empty:
        raise ValueError(
            "Duplicate stock codes found in product catalogue."
        )

    # Check for missing product identifiers.
    if catalog["stock_code"].isna().any():
        raise ValueError(
            "Missing stock codes found in product catalogue."
        )

    # Check the feature dataset identifiers.
    df["stock_code"] = df["stock_code"].str.strip()

    products_before = df["stock_code"].nunique()
    rows_before = len(df)

    print(
        f"Products in feature dataset: {products_before:,}"
    )

    print(
        f"Products in catalogue: {len(catalog):,}"
    )

    # Merge product descriptions into the feature dataset.
    df = df.merge(
        catalog,
        on="stock_code",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    # Find products that failed to match.
    unmatched = df.loc[
        df["_merge"] != "both",
        "stock_code"
    ].drop_duplicates()

    if not unmatched.empty:

        print("\nUnmatched products:")
        print(unmatched.head(20).to_list())

        raise ValueError(
            f"{len(unmatched)} products could not be matched."
        )

    # Remove the temporary merge indicator.
    df = df.drop(columns=["_merge"])
    
    # Identify missing descriptions before filling them.
    df["description_missing"] = (
        df["description"].isna().astype("int8")
    )

    # Count affected product-day rows.
    missing_description_rows = int(
        df["description_missing"].sum()
    )

    # Count affected unique products.
    missing_description_products = (
        df.loc[
            df["description_missing"] == 1,
            "stock_code"
        ].nunique()
    )

    # Replace missing descriptions with a clear label.
    df["description"] = df["description"].fillna("Unknown")

    print(
        "Products with missing descriptions:",
        missing_description_products
    )

    print(
        "Product-day rows with missing descriptions:",
        missing_description_rows
    )

    print("Missing descriptions handled successfully.")



    # Verify that the merge did not multiply rows.
    if len(df) != rows_before:
        raise ValueError(
            "Row count changed after product merge."
        )

    # Report results.
    print("\nProduct feature merge successful.")

    print(
        f"Rows preserved: {len(df):,}"
    )

    print(
        f"Products matched: {products_before:,}"
    )



    return df

# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

def main():

    print("=" * 55)
    print("DEMANDIQ - DAY 5 FEATURE ENGINEERING")
    print("=" * 55)

    df = load_panel(PANEL_PATH)

    df = df.sort_values(
        ["stock_code", "sales_date"]
    ).reset_index(drop=True)

    validate_input(df)
     # Product information
    df = add_product_features(df)

    # Calendar features
    df = add_calendar_features(df)

    # Lag features
    df = add_lag_features(df)

    # Rolling mean features
    df = add_rolling_features(df)

    # Rolling standard deviation features
    df = add_rolling_volatility_features(df)

    # Display important feature columns
    feature_columns = [
    "stock_code",
    "description",
    "description_missing",
    "sales_date",
    "daily_units_sold",

    # Calendar features
    "day_of_week",
    "month",
    "quarter",
    "is_weekend",
    "is_saturday",
    "is_sunday",

    # Lag features
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",

    # Rolling means
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",

    # Rolling standard deviations
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
]

    print("\nFEATURE DATASET PREVIEW")
    print("-" * 55)

    print(
        df[feature_columns].head(40).to_string(index=False)
    )

    print("\nMISSING FEATURE VALUES")
    print("-" * 55)

    print(
        df[feature_columns[3:]].isna().sum()
    )

    print("\nDATASET SUMMARY")
    print("-" * 55)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nSTEP 3 AND STEP 4 COMPLETED SUCCESSFULLY.")

    # --------------------------------------------------
    # SAVE ENGINEERED DATASET AND REPORT
    # --------------------------------------------------

    print("\nSaving engineered dataset...")

    # Ensure output directories exist.
    FEATURE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Generate feature summary.
    summary = pd.DataFrame({
        "metric": [
            "total_rows",
            "total_columns",
            "unique_products",
            "first_date",
            "last_date",
            "description_missing_products",
            "description_missing_rows",
            "lag_features",
            "rolling_mean_features",
            "rolling_std_features",
        ],

        "value": [
            len(df),
            len(df.columns),
            df["stock_code"].nunique(),
            str(df["sales_date"].min().date()),
            str(df["sales_date"].max().date()),
            df.loc[
                df["description_missing"] == 1,
                "stock_code"
            ].nunique(),
            int(df["description_missing"].sum()),
            4,
            3,
            3,
        ],
    })

    # Save feature dataset.
    df.to_csv(
        FEATURE_OUTPUT_PATH,
        index=False
    )

    # Save feature summary.
    summary.to_csv(
        REPORT_PATH,
        index=False
    )

    print("Feature dataset saved:")
    print(FEATURE_OUTPUT_PATH)

    print("\nFeature summary saved:")
    print(REPORT_PATH)

    print("\nDAY 5 FEATURE ENGINEERING COMPLETED.")


if __name__ == "__main__":
    main()
