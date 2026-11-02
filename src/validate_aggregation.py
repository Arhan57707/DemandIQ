
from pathlib import Path
import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

PRIMARY_FILE = DATA_DIR / "retail_demand_clean.csv"

SPARSE_FILE = DATA_DIR / "daily_product_sales.csv"

PANEL_FILE = DATA_DIR / "daily_product_panel.csv"

CATALOG_FILE = DATA_DIR / "product_catalog.csv"


# ==================================================
# 2. LOAD DATA
# ==================================================

def validate_aggregation():

    print("=" * 55)
    print("DEMANDIQ - AGGREGATION VALIDATION")
    print("=" * 55)

    # Check that all required files exist
    required_files = [
        PRIMARY_FILE,
        SPARSE_FILE,
        PANEL_FILE,
        CATALOG_FILE
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    print("\nLoading datasets...")

    # Stock codes are identifiers, so always read them
    # as strings instead of allowing numeric inference.

    primary = pd.read_csv(
        PRIMARY_FILE,
        usecols=["quantity", "stock_code"],
        dtype={"stock_code": "string"}
    )

    sparse = pd.read_csv(
        SPARSE_FILE,
        dtype={"stock_code": "string"},
        parse_dates=["sales_date"],
        low_memory=False
    )

    panel = pd.read_csv(
        PANEL_FILE,
        usecols=[
            "stock_code",
            "sales_date",
            "daily_units_sold",
            "has_transaction"
        ],
        dtype={"stock_code": "string"},
        parse_dates=["sales_date"],
        low_memory=False
    )

    catalog = pd.read_csv(
        CATALOG_FILE,
        dtype={"stock_code": "string"},
        low_memory=False
    )

    # Standardise product identifiers
    # to avoid whitespace differences.

    for df in [primary, sparse, panel, catalog]:

        df["stock_code"] = (
            df["stock_code"].str.strip()
        )

    # ==================================================
    # 3. UNIT RECONCILIATION
    # ==================================================

    source_units = primary["quantity"].sum()

    sparse_units = sparse[
        "daily_units_sold"
    ].sum()

    panel_units = panel[
        "daily_units_sold"
    ].sum()

    print("\nUNIT RECONCILIATION")
    print("-" * 55)

    print("Clean transaction units:", source_units)
    print("Sparse daily units:", sparse_units)
    print("Dense panel units:", panel_units)

    check_units = (
        source_units == sparse_units == panel_units
    )

    # ==================================================
    # 4. SPARSE DATA VALIDATION
    # ==================================================

    sparse_unique = not sparse.duplicated(
        subset=["stock_code", "sales_date"]
    ).any()

    sparse_positive = (
        sparse["daily_units_sold"] > 0
    ).all()

    # ==================================================
    # 5. DENSE PANEL VALIDATION
    # ==================================================

    panel_unique = not panel.duplicated(
        subset=["stock_code", "sales_date"]
    ).any()

    panel_nonnegative = (
        panel["daily_units_sold"] >= 0
    ).all()

    flags_valid = (
        panel["has_transaction"]
        .isin([0, 1])
        .all()
    )

    flags_consistent = (
        (
            (panel["daily_units_sold"] > 0)
            ==
            (panel["has_transaction"] == 1)
        ).all()
    )

    # ==================================================
    # 6. PRODUCT COVERAGE VALIDATION
    # ==================================================

    sparse_codes = set(
        sparse["stock_code"].dropna().unique()
    )

    catalog_codes = set(
        catalog["stock_code"].dropna().unique()
    )

    panel_codes = set(
        panel["stock_code"].dropna().unique()
    )

    product_coverage = (
        sparse_codes == catalog_codes == panel_codes
    )

    print("\nPRODUCT COVERAGE INVESTIGATION")
    print("-" * 55)

    print("Sparse dataset products:", len(sparse_codes))
    print("Product catalogue products:", len(catalog_codes))
    print("Dense panel products:", len(panel_codes))

    print(
        "Missing from catalogue:",
        len(sparse_codes - catalog_codes)
    )

    print(
        "Missing from dense panel:",
        len(sparse_codes - panel_codes)
    )

    print(
        "Extra in catalogue:",
        len(catalog_codes - sparse_codes)
    )

    print(
        "Extra in dense panel:",
        len(panel_codes - sparse_codes)
    )

    # Display examples when product coverage fails
    if not product_coverage:

        print("\nProducts missing from catalogue:")
        print(
            sorted(sparse_codes - catalog_codes)[:20]
        )

        print("\nProducts missing from dense panel:")
        print(
            sorted(sparse_codes - panel_codes)[:20]
        )

    # ==================================================
    # 7. PRODUCT CATALOGUE VALIDATION
    # ==================================================

    catalog_unique = not catalog[
        "stock_code"
    ].duplicated().any()

    # ==================================================
    # 8. PRINT ALL VALIDATION RESULTS
    # ==================================================

    checks = {

        "Total units preserved across all datasets":
            check_units,

        "No duplicate product-date rows in sparse data":
            sparse_unique,

        "All sparse daily quantities are positive":
            sparse_positive,

        "No duplicate product-date rows in dense panel":
            panel_unique,

        "No negative quantities in dense panel":
            panel_nonnegative,

        "Transaction flags contain only 0 and 1":
            flags_valid,

        "Transaction flags match recorded sales":
            flags_consistent,

        "Product coverage matches across datasets":
            product_coverage,

        "Product catalogue has unique stock codes":
            catalog_unique
    }

    print("\nVALIDATION RESULTS")
    print("-" * 55)

    all_passed = True

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status}: {name}")

        if not passed:
            all_passed = False

    # ==================================================
    # 9. PANEL SUMMARY
    # ==================================================

    print("\nPANEL SUMMARY")

    print(
        "Panel rows:",
        f"{len(panel):,}"
    )

    print(
        "Zero-sales product-days:",
        f"{(panel['daily_units_sold'] == 0).sum():,}"
    )

    print(
        "Products:",
        f"{panel['stock_code'].nunique():,}"
    )

    # ==================================================
    # 10. FINAL RESULT
    # ==================================================

    if not all_passed:

        raise ValueError(
            "One or more aggregation validation checks failed."
        )

    print("\nAll aggregation checks passed!")


if __name__ == "__main__":
    validate_aggregation()
