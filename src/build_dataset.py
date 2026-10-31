
from pathlib import Path
import pandas as pd


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


# Original column names -> standard project column names
RENAME_COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country"
}


def build_dataset():

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {RAW_FILE}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all Excel worksheets...")

    # Load all worksheets into a dictionary
    worksheets = pd.read_excel(
        RAW_FILE,
        sheet_name=None
    )

    frames = []

    # Standardise column names and combine worksheets
    for sheet_name, frame in worksheets.items():

        print(f"Reading worksheet: {sheet_name}")

        frame.columns = frame.columns.astype(str).str.strip()

        frame = frame.rename(
            columns=RENAME_COLUMNS
        )

        # Keep track of the worksheet each row came from
        frame["source_sheet"] = sheet_name

        frames.append(frame)

    # Combine all worksheets
    df = pd.concat(
        frames,
        ignore_index=True
    )

    # Check required columns
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
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    print("\nConverting data types...")

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

    # Calculate the transaction line value.
    # Negative values can represent returns or cancellations.
    df["line_value"] = df["quantity"] * df["unit_price"]

    print("\nCreating data quality report...")

    # Column-level quality report
    quality_report = pd.DataFrame(index=df.columns)

    quality_report["data_type"] = df.dtypes.astype(str)

    quality_report["missing_count"] = df.isnull().sum()

    quality_report["missing_percent"] = (
        df.isnull().mean() * 100
    ).round(2)

    quality_report["unique_values"] = (
        df.nunique(dropna=True)
    )

    quality_report = quality_report.sort_values(
        by="missing_percent",
        ascending=False
    )

    # Count exact duplicate transactions without using source_sheet
    duplicate_count = int(
        df.drop(columns=["source_sheet"])
        .duplicated()
        .sum()
    )

    # Additional data quality indicators
    cancelled_invoice_count = int(
        df["invoice"]
        .astype("string")
        .str.startswith("C", na=False)
        .sum()
    )

    negative_quantity_count = int(
        (df["quantity"] < 0).sum()
    )

    nonpositive_price_count = int(
        (df["unit_price"] <= 0).sum()
    )

    invalid_date_count = int(
        df["invoice_date"].isna().sum()
    )

    missing_customer_count = int(
        df["customer_id"].isna().sum()
    )

    metrics = pd.DataFrame({
        "metric": [
            "total_rows",
            "total_columns",
            "exact_duplicate_rows",
            "cancelled_invoice_rows",
            "negative_quantity_rows",
            "nonpositive_price_rows",
            "invalid_invoice_dates",
            "missing_customer_ids"
        ],
        "value": [
            len(df),
            len(df.columns),
            duplicate_count,
            cancelled_invoice_count,
            negative_quantity_count,
            nonpositive_price_count,
            invalid_date_count,
            missing_customer_count
        ]
    })

    # Save reports
    quality_report.to_csv(
        REPORTS_DIR / "data_quality_summary.csv"
    )

    metrics.to_csv(
        REPORTS_DIR / "data_quality_metrics.csv",
        index=False
    )

    # Save combined dataset locally
    output_file = (
        PROCESSED_DIR / "retail_transactions_combined.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 55)
    print("DEMANDIQ DATASET BUILD COMPLETED")
    print("=" * 55)

    print(f"\nTotal rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print("\nWorksheet distribution:")
    print(df["source_sheet"].value_counts().to_string())

    print("\nData quality metrics:")
    print(metrics.to_string(index=False))

    print("\nSaved files:")
    print(output_file)
    print(REPORTS_DIR / "data_quality_summary.csv")
    print(REPORTS_DIR / "data_quality_metrics.csv")


if __name__ == "__main__":
    build_dataset()
