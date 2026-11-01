
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retail_transactions_combined.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

REPORT_DIR = PROJECT_ROOT / "reports"


# Original business columns used for exact duplicate checks
DUPLICATE_COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country"
]


# --------------------------------------------------
# 2. DATA LOADING AND STANDARDISATION
# --------------------------------------------------

def load_data():

    print("Loading combined retail dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={
            "invoice": "string",
            "stock_code": "string"
        }
    )

    required_columns = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    # Clean text columns
    for column in [
        "invoice",
        "stock_code",
        "description",
        "country"
    ]:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        df[column] = df[column].replace(
            "",
            pd.NA
        )

    # Convert date and numeric fields safely
    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"],
        errors="coerce"
    )

    df["quantity"] = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    df["unit_price"] = pd.to_numeric(
        df["unit_price"],
        errors="coerce"
    )

    # Recalculate transaction line value
    df["line_value"] = (
        df["quantity"] * df["unit_price"]
    )

    return df


# --------------------------------------------------
# 3. APPLY THE DEMAND CLEANING RULES
# --------------------------------------------------

def prepare_demand_data(data):

    df = data.copy()

    original_rows = len(df)

    # Identify records that belong to duplicate groups
    duplicate_group_mask = df.duplicated(
        subset=DUPLICATE_COLUMNS,
        keep=False
    )

    duplicate_group_rows = int(
        duplicate_group_mask.sum()
    )

    # Count extra copies that would be removed
    exact_duplicate_copies = int(
        df.duplicated(
            subset=DUPLICATE_COLUMNS,
            keep="first"
        ).sum()
    )

    # Keep a review copy of all records in duplicate groups
    duplicate_records = df[
        duplicate_group_mask
    ].copy()

    # Remove extra copies for the primary dataset
    deduplicated = df.drop_duplicates(
        subset=DUPLICATE_COLUMNS,
        keep="first"
    ).copy()

    after_deduplication = len(deduplicated)

    # Identify essential fields
    valid_required_fields = (
        deduplicated["invoice"].notna()
        & deduplicated["stock_code"].notna()
        & deduplicated["invoice_date"].notna()
        & deduplicated["quantity"].notna()
    )

    invalid_required_records = deduplicated[
        ~valid_required_fields
    ].copy()

    valid_records = deduplicated[
        valid_required_fields
    ].copy()

    invalid_required_count = (
        len(invalid_required_records)
    )

    # Identify cancelled invoice identifiers
    cancelled_mask = (
        valid_records["invoice"]
        .str.upper()
        .str.startswith("C", na=False)
    )

    cancelled_records = valid_records[
        cancelled_mask
    ].copy()

    non_cancelled = valid_records[
        ~cancelled_mask
    ].copy()

    cancelled_count = len(cancelled_records)

    # Identify zero and negative quantities
    nonpositive_quantity_mask = (
        non_cancelled["quantity"] <= 0
    )

    nonpositive_quantity_records = (
        non_cancelled[
            nonpositive_quantity_mask
        ].copy()
    )

    positive_demand = non_cancelled[
        ~nonpositive_quantity_mask
    ].copy()

    nonpositive_quantity_count = (
        len(nonpositive_quantity_records)
    )

    # Add useful analysis flags
    positive_demand["revenue_eligible"] = (
        positive_demand["unit_price"] > 0
    )

    positive_demand["line_value"] = (
        positive_demand["quantity"]
        * positive_demand["unit_price"]
    )

    # Return records are kept separately for future analysis.
    # This review file is based on the deduplicated data.
    returns_review = deduplicated[
        deduplicated["quantity"] < 0
    ].copy()

    # The version retaining duplicate rows is useful
    # for comparing how duplicate removal affects demand.
    return {
        "clean": positive_demand,
        "duplicates": duplicate_records,
        "returns": returns_review,
        "invalid": invalid_required_records,
        "cancelled": cancelled_records,
        "nonpositive_quantity": nonpositive_quantity_records,
        "metrics": {
            "original_rows": original_rows,
            "duplicate_group_rows": duplicate_group_rows,
            "exact_duplicate_copies_removed": exact_duplicate_copies,
            "rows_after_deduplication": after_deduplication,
            "invalid_required_records_removed": invalid_required_count,
            "cancelled_records_removed": cancelled_count,
            "nonpositive_quantity_records_removed": nonpositive_quantity_count,
            "final_clean_rows": len(positive_demand)
        }
    }


# --------------------------------------------------
# 4. CREATE AN ALTERNATIVE VERSION
#    THAT RETAINS DUPLICATES
# --------------------------------------------------

def create_duplicate_alternative(data):

    df = data.copy()

    valid_required_fields = (
        df["invoice"].notna()
        & df["stock_code"].notna()
        & df["invoice_date"].notna()
        & df["quantity"].notna()
    )

    df = df[valid_required_fields].copy()

    cancelled_mask = (
        df["invoice"]
        .str.upper()
        .str.startswith("C", na=False)
    )

    df = df[~cancelled_mask].copy()

    df = df[df["quantity"] > 0].copy()

    df["revenue_eligible"] = (
        df["unit_price"] > 0
    )

    df["line_value"] = (
        df["quantity"] * df["unit_price"]
    )

    return df


# --------------------------------------------------
# 5. GENERATE CLEANING REPORT
# --------------------------------------------------

def create_report(data, results, alternative):

    metrics = results["metrics"]

    clean = results["clean"]

    report_values = [
        ("original_rows", metrics["original_rows"],
         "Rows in the combined input dataset"),

        ("duplicate_group_rows", metrics["duplicate_group_rows"],
         "Rows belonging to exact duplicate groups"),

        ("exact_duplicate_copies_removed",
         metrics["exact_duplicate_copies_removed"],
         "Extra copies removed in the primary dataset"),

        ("rows_after_deduplication",
         metrics["rows_after_deduplication"],
         "Rows remaining after exact deduplication"),

        ("invalid_required_records_removed",
         metrics["invalid_required_records_removed"],
         "Missing essential transaction fields"),

        ("cancelled_records_removed",
         metrics["cancelled_records_removed"],
         "Valid records with invoice identifiers starting with C"),

        ("nonpositive_quantity_records_removed",
         metrics["nonpositive_quantity_records_removed"],
         "Zero or negative quantities excluded from positive demand"),

        ("final_clean_rows",
         metrics["final_clean_rows"],
         "Rows retained in the primary positive-demand dataset"),

        ("alternative_rows_retaining_duplicates",
         len(alternative),
         "Positive-demand rows before exact deduplication"),

        ("negative_quantity_review_rows",
         len(results["returns"]),
         "Negative quantity rows retained for separate review"),

        ("missing_customer_ids_retained",
         int(clean["customer_id"].isna().sum()),
         "Customer IDs missing in the final clean dataset"),

        ("missing_descriptions_retained",
         int(clean["description"].isna().sum()),
         "Descriptions missing in the final clean dataset"),

        ("nonpositive_price_rows_retained",
         int((clean["unit_price"] <= 0).sum()),
         "Rows retained for unit demand, flagged as revenue-ineligible")
    ]

    report = pd.DataFrame(
        report_values,
        columns=[
            "metric",
            "value",
            "explanation"
        ]
    )

    return report


# --------------------------------------------------
# 6. VALIDATE THE CLEAN DATASET
# --------------------------------------------------

def validate_data(clean):

    print("\nVALIDATING CLEAN DATASET")
    print("-" * 50)

    checks = {
        "No missing invoice IDs":
            clean["invoice"].notna().all(),

        "No missing stock codes":
            clean["stock_code"].notna().all(),

        "No invalid invoice dates":
            clean["invoice_date"].notna().all(),

        "All quantities are positive":
            (clean["quantity"] > 0).all(),

        "No cancelled invoice identifiers":
            (~clean["invoice"]
             .str.upper()
             .str.startswith("C", na=False)).all(),

        "No exact duplicate business records":
            not clean.duplicated(
                subset=DUPLICATE_COLUMNS
            ).any()
    }

    all_passed = True

    for check_name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status}: {check_name}")

        if not passed:
            all_passed = False

    if not all_passed:
        raise ValueError(
            "One or more data validation checks failed."
        )

    print("\nAll validation checks passed!")


# --------------------------------------------------
# 7. SAVE THE RESULTS
# --------------------------------------------------

def save_outputs(results, alternative, report):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Main dataset used for later demand analysis
    results["clean"].to_csv(
        OUTPUT_DIR / "retail_demand_clean.csv",
        index=False
    )

    # Alternative dataset retaining exact duplicates
    alternative.to_csv(
        OUTPUT_DIR / "retail_demand_clean_with_duplicates.csv",
        index=False
    )

    # Separate negative quantity review file
    results["returns"].to_csv(
        OUTPUT_DIR / "retail_returns_review.csv",
        index=False
    )

    # Save the full duplicate review file locally
    results["duplicates"].to_csv(
        OUTPUT_DIR / "duplicate_records_review.csv",
        index=False
    )

    # Save the cleaning report
    report.to_csv(
        REPORT_DIR / "data_cleaning_summary.csv",
        index=False
    )

    print("\nOUTPUT FILES SAVED")
    print("-" * 50)

    print(OUTPUT_DIR / "retail_demand_clean.csv")
    print(OUTPUT_DIR / "retail_demand_clean_with_duplicates.csv")
    print(OUTPUT_DIR / "retail_returns_review.csv")
    print(OUTPUT_DIR / "duplicate_records_review.csv")
    print(REPORT_DIR / "data_cleaning_summary.csv")


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

def main():

    print("=" * 60)
    print("DEMANDIQ - DATA CLEANING PIPELINE")
    print("=" * 60)

    data = load_data()

    print("\nOriginal rows:", len(data))

    results = prepare_demand_data(data)

    alternative = create_duplicate_alternative(data)

    report = create_report(
        data,
        results,
        alternative
    )

    validate_data(results["clean"])

    save_outputs(
        results,
        alternative,
        report
    )

    print("\n" + "=" * 60)
    print("DEMANDIQ CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        "\nOriginal rows:",
        len(data)
    )

    print(
        "Final clean rows:",
        len(results["clean"])
    )

    print(
        "\nRows retained:",
        round(
            len(results["clean"]) / len(data) * 100,
            2
        ),
        "%"
    )

    print(
        "\nRows excluded from primary dataset:",
        len(data) - len(results["clean"])
    )


if __name__ == "__main__":
    main()
