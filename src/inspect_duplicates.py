
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retail_transactions_combined.csv"
)


def inspect_duplicates():

    print("Loading transaction dataset...")

    df = pd.read_csv(DATA_FILE)

    # These are the original business transaction fields.
    # We exclude generated columns such as source_sheet
    # and line_value from the duplicate comparison.
    duplicate_columns = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country"
    ]

    duplicate_mask = df.duplicated(
        subset=duplicate_columns,
        keep=False
    )

    duplicate_records = df[duplicate_mask].copy()

    exact_duplicate_count = df.duplicated(
        subset=duplicate_columns
    ).sum()

    print("\nDUPLICATE INVESTIGATION")
    print("-" * 50)

    print("Total dataset rows:", len(df))

    print(
        "Rows belonging to duplicate groups:",
        len(duplicate_records)
    )

    print(
        "Extra duplicate copies:",
        exact_duplicate_count
    )

    print("\nEXAMPLES OF DUPLICATE RECORDS:")

    print(
        duplicate_records[
            duplicate_columns + ["source_sheet"]
        ].head(20).to_string(index=False)
    )

    print("\nDUPLICATES BY WORKSHEET:")

    print(
        duplicate_records["source_sheet"]
        .value_counts()
        .to_string()
    )

    # Save a review copy locally.
    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "duplicate_records_review.csv"
    )

    duplicate_records.to_csv(
        output_file,
        index=False
    )

    print("\nDuplicate review file saved:")
    print(output_file)


if __name__ == "__main__":
    inspect_duplicates()
