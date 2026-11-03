
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# DEMANDIQ - DAY 5 FEATURE VALIDATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "daily_product_features.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "feature_validation_summary.csv"
)

CHUNK_SIZE = 200_000

EXPECTED_ROWS = 3_084_067
EXPECTED_PRODUCTS = 4_984
EXPECTED_COLUMNS = 37
EXPECTED_TOTAL_UNITS = 11_455_912

EXPECTED_MISSING_DESCRIPTION_ROWS = 44_763
EXPECTED_MISSING_DESCRIPTION_PRODUCTS = 64


# --------------------------------------------------
# 1. REQUIRED COLUMNS
# --------------------------------------------------

REQUIRED_COLUMNS = [
    "stock_code",
    "description",
    "description_missing",
    "sales_date",
    "daily_units_sold",
    "has_transaction",

    "day_of_week",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "year",

    "is_weekend",
    "is_saturday",
    "is_sunday",
    "is_december",

    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",

    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
]


# --------------------------------------------------
# 2. VALIDATION RESULT TRACKER
# --------------------------------------------------

results = []


def add_result(name, passed, observed):

    status = "PASS" if passed else "FAIL"

    results.append({
        "validation": name,
        "status": status,
        "observed": str(observed),
    })

    print(
        f"{status}: {name} | {observed}"
    )


# --------------------------------------------------
# 3. CHECK FILE AND COLUMN STRUCTURE
# --------------------------------------------------

def validate_file_structure():

    print("\nCHECKING FILE STRUCTURE")
    print("-" * 55)

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_PATH}"
        )

    header = pd.read_csv(
        FEATURE_PATH,
        nrows=0
    )

    columns = header.columns.tolist()

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in columns
    ]

    add_result(
        "Expected total columns",
        len(columns) == EXPECTED_COLUMNS,
        len(columns),
    )

    add_result(
        "Required feature columns exist",
        len(missing_columns) == 0,
        missing_columns if missing_columns else "All present",
    )

    selected_columns = [
        col for col in REQUIRED_COLUMNS
        if col in columns
    ]

    return selected_columns


# --------------------------------------------------
# 4. CHUNKED DATASET VALIDATION
# --------------------------------------------------

def validate_dataset(columns):

    print("\nVALIDATING FEATURE DATASET")
    print("-" * 55)

    total_rows = 0
    total_units = 0

    products_seen = set()

    missing_description_rows = 0
    missing_description_products = set()

    actual_missing = {}

    for column in [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_14",
        "rolling_std_28",
    ]:
        actual_missing[column] = 0

    product_row_counts = {}

    negative_features = 0
    duplicate_rows = 0
    sorted_order = True
    previous_key = None

    invalid_descriptions = 0
    invalid_flags = 0
    inconsistent_description_flags = 0

    calendar_mismatches = 0
    weekend_mismatches = 0
    saturday_mismatches = 0
    sunday_mismatches = 0
    december_mismatches = 0

    transaction_flag_mismatches = 0
    invalid_cyclical_values = 0

    non_finite_features = 0

    feature_columns = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_14",
        "rolling_std_28",
    ]

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            FEATURE_PATH,
            usecols=columns,
            dtype={
                "stock_code": "string",
                "description": "string",
            },
            parse_dates=["sales_date"],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        total_rows += len(chunk)

        # ------------------------------------------
        # Basic statistics
        # ------------------------------------------

        total_units += int(
            chunk["daily_units_sold"].sum()
        )

        products_seen.update(
            chunk["stock_code"].dropna().unique()
        )

        # ------------------------------------------
        # Product row counts
        # ------------------------------------------

        counts = chunk.groupby(
            "stock_code"
        ).size()

        for stock_code, count in counts.items():

            product_row_counts[stock_code] = (
                product_row_counts.get(stock_code, 0)
                + int(count)
            )

        # ------------------------------------------
        # Missing description validation
        # ------------------------------------------

        description_missing = (
            chunk["description_missing"] == 1
        )

        missing_description_rows += int(
            description_missing.sum()
        )

        missing_description_products.update(
            chunk.loc[
                description_missing,
                "stock_code"
            ].dropna().unique()
        )

        invalid_descriptions += int(
            (
                chunk["description"].isna()
                | chunk["description"].str.strip().eq("")
            ).sum()
        )

        invalid_flags += int(
            (
                ~chunk["description_missing"].isin([0, 1])
            ).sum()
        )

        # A missing-description flag should be
        # consistent for every date of a product.
        for stock_code, group in chunk.groupby("stock_code"):

            if group["description_missing"].nunique() > 1:
                inconsistent_description_flags += 1

        # ------------------------------------------
        # Lag and rolling missing values
        # ------------------------------------------

        for column in feature_columns:

            actual_missing[column] += int(
                chunk[column].isna().sum()
            )

            non_missing = chunk[column].dropna()

            negative_features += int(
                (non_missing < 0).sum()
            )

            non_finite_features += int(
                (~np.isfinite(non_missing)).sum()
            )

        # ------------------------------------------
        # Calendar validation
        # ------------------------------------------

        dates = chunk["sales_date"]

        expected_day = dates.dt.day_of_week

        calendar_columns = {
            "day_of_week": expected_day,
            "day_of_month": dates.dt.day,
            "day_of_year": dates.dt.day_of_year,
            "week_of_year": dates.dt.isocalendar().week.astype("int16"),
            "month": dates.dt.month,
            "quarter": dates.dt.quarter,
            "year": dates.dt.year,
        }

        for column, expected in calendar_columns.items():

            calendar_mismatches += int(
                (chunk[column].to_numpy() != expected.to_numpy()).sum()
            )

        weekend_mismatches += int(
            (
                chunk["is_weekend"].to_numpy()
                != (expected_day >= 5).astype("int8").to_numpy()
            ).sum()
        )

        saturday_mismatches += int(
            (
                chunk["is_saturday"].to_numpy()
                != (expected_day == 5).astype("int8").to_numpy()
            ).sum()
        )

        sunday_mismatches += int(
            (
                chunk["is_sunday"].to_numpy()
                != (expected_day == 6).astype("int8").to_numpy()
            ).sum()
        )

        december_mismatches += int(
            (
                chunk["is_december"].to_numpy()
                != (dates.dt.month == 12).astype("int8").to_numpy()
            ).sum()
        )

        # ------------------------------------------
        # Transaction flag validation
        # ------------------------------------------

        expected_transaction = (
            chunk["daily_units_sold"] > 0
        ).astype("int8")

        transaction_flag_mismatches += int(
            (
                chunk["has_transaction"].to_numpy()
                != expected_transaction.to_numpy()
            ).sum()
        )

        # ------------------------------------------
        # Cyclical feature validation
        # ------------------------------------------

        expected_dow_sin = np.sin(
            2 * np.pi * expected_day / 7
        )

        expected_dow_cos = np.cos(
            2 * np.pi * expected_day / 7
        )

        expected_month_sin = np.sin(
            2 * np.pi * (dates.dt.month - 1) / 12
        )

        expected_month_cos = np.cos(
            2 * np.pi * (dates.dt.month - 1) / 12
        )

        for column, expected in [
            ("day_of_week_sin", expected_dow_sin),
            ("day_of_week_cos", expected_dow_cos),
            ("month_sin", expected_month_sin),
            ("month_cos", expected_month_cos),
        ]:

            invalid_cyclical_values += int(
                (
                    ~np.isclose(
                        chunk[column].to_numpy(),
                        expected.to_numpy(),
                        atol=1e-9,
                        rtol=1e-9,
                    )
                ).sum()
            )

        # ------------------------------------------
        # Duplicate product-date validation
        # ------------------------------------------

        duplicate_rows += int(
            chunk.duplicated(
                subset=["stock_code", "sales_date"]
            ).sum()
        )

        # The dataset should remain sorted by
        # product and date across all chunks.
        keys = pd.MultiIndex.from_arrays([
            chunk["stock_code"],
            chunk["sales_date"],
        ])

        if not keys.is_monotonic_increasing:
            sorted_order = False

        first_key = (
            str(chunk["stock_code"].iloc[0]),
            chunk["sales_date"].iloc[0],
        )

        last_key = (
            str(chunk["stock_code"].iloc[-1]),
            chunk["sales_date"].iloc[-1],
        )

        if previous_key is not None:

            if first_key < previous_key:
                sorted_order = False

            if first_key == previous_key:
                duplicate_rows += 1

        previous_key = last_key

        if chunk_number % 5 == 0:
            print(
                f"Processed {total_rows:,} rows..."
            )

    # --------------------------------------------------
    # 5. RECORD VALIDATION RESULTS
    # --------------------------------------------------

    print("\nSTRUCTURAL VALIDATION")
    print("-" * 55)

    add_result(
        "Total row count",
        total_rows == EXPECTED_ROWS,
        total_rows,
    )

    add_result(
        "Unique product count",
        len(products_seen) == EXPECTED_PRODUCTS,
        len(products_seen),
    )

    add_result(
        "Total demand preserved",
        total_units == EXPECTED_TOTAL_UNITS,
        total_units,
    )

    print("\nPRODUCT VALIDATION")
    print("-" * 55)

    add_result(
        "Description missing rows handled",
        invalid_descriptions == 0,
        invalid_descriptions,
    )

    add_result(
        "Description missingness flag count",
        missing_description_rows
        == EXPECTED_MISSING_DESCRIPTION_ROWS,
        missing_description_rows,
    )

    add_result(
        "Products with missing descriptions",
        len(missing_description_products)
        == EXPECTED_MISSING_DESCRIPTION_PRODUCTS,
        len(missing_description_products),
    )

    add_result(
        "Description flags contain only 0 and 1",
        invalid_flags == 0,
        invalid_flags,
    )

    add_result(
        "Description flag consistency",
        inconsistent_description_flags == 0,
        inconsistent_description_flags,
    )

    print("\nDUPLICATE VALIDATION")
    print("-" * 55)

    add_result(
        "Dataset sorted by product and date",
        sorted_order,
        sorted_order,
    )

    add_result(
        "No duplicate product-date rows",
        duplicate_rows == 0,
        duplicate_rows,
    )

    print("\nCALENDAR VALIDATION")
    print("-" * 55)

    add_result(
        "Calendar values match dates",
        calendar_mismatches == 0,
        calendar_mismatches,
    )

    add_result(
        "Weekend flags correct",
        weekend_mismatches == 0,
        weekend_mismatches,
    )

    add_result(
        "Saturday flags correct",
        saturday_mismatches == 0,
        saturday_mismatches,
    )

    add_result(
        "Sunday flags correct",
        sunday_mismatches == 0,
        sunday_mismatches,
    )

    add_result(
        "December flags correct",
        december_mismatches == 0,
        december_mismatches,
    )

    add_result(
        "Cyclical features correct",
        invalid_cyclical_values == 0,
        invalid_cyclical_values,
    )

    print("\nDEMAND FEATURE VALIDATION")
    print("-" * 55)

    add_result(
        "Transaction flags match recorded demand",
        transaction_flag_mismatches == 0,
        transaction_flag_mismatches,
    )

    add_result(
        "Lag and rolling features are non-negative",
        negative_features == 0,
        negative_features,
    )

    add_result(
        "Lag and rolling features contain no infinity",
        non_finite_features == 0,
        non_finite_features,
    )

    # ------------------------------------------
    # Check expected historical warm-up NaNs
    # ------------------------------------------

    expected_missing = {}

    # Calculate expected missing values for lag features.
    for window in [1, 7, 14, 28]:

        expected_count = sum(
            min(product_days, window)
            for product_days in product_row_counts.values()
        )

        expected_missing[f"lag_{window}"] = expected_count

    # Rolling features only use 7, 14 and 28-day windows.
    for window in [7, 14, 28]:

        expected_count = sum(
            min(product_days, window)
            for product_days in product_row_counts.values()
        )

        expected_missing[f"rolling_mean_{window}"] = expected_count

        expected_missing[f"rolling_std_{window}"] = expected_count

    print("\nHISTORICAL FEATURE MISSING VALUES")
    print("-" * 55)

    for column, expected in expected_missing.items():

        actual = actual_missing[column]

        add_result(
            f"{column} expected missing values",
            actual == expected,
            f"actual={actual:,}, expected={expected:,}",
        )

    # --------------------------------------------------
    # 6. SAVE VALIDATION REPORT
    # --------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = pd.DataFrame(results)

    report.to_csv(
        REPORT_PATH,
        index=False,
    )

    passed = sum(
        result["status"] == "PASS"
        for result in results
    )

    failed = len(results) - passed

    print("\n" + "=" * 55)
    print("FEATURE VALIDATION SUMMARY")
    print("=" * 55)

    print("Total checks:", len(results))
    print("Passed:", passed)
    print("Failed:", failed)

    print("\nReport saved:")
    print(REPORT_PATH)

    if failed == 0:
        print("\nALL FEATURE VALIDATION CHECKS PASSED!")
    else:
        print("\nSOME VALIDATION CHECKS FAILED.")
        raise AssertionError(
            "Feature dataset validation failed."
        )


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

def main():

    print("=" * 55)
    print("DEMANDIQ - DAY 5 FEATURE VALIDATION")
    print("=" * 55)

    columns = validate_file_structure()

    validate_dataset(columns)


if __name__ == "__main__":
    main()
