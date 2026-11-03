
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "product_catalog.csv"
)

PANEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "daily_product_panel.csv"
)


def main():

    print("=" * 55)
    print("DEMANDIQ - MISSING PRODUCT DESCRIPTION INVESTIGATION")
    print("=" * 55)

    # Load product catalogue.
    catalog = pd.read_csv(
        CATALOG_PATH,
        dtype={"stock_code": "string"},
    )

    # Remove surrounding whitespace.
    catalog["stock_code"] = (
        catalog["stock_code"].str.strip()
    )

    catalog["description"] = (
        catalog["description"].astype("string").str.strip()
    )

    # Identify products without descriptions.
    missing_catalog = catalog[
        catalog["description"].isna()
        | catalog["description"].eq("")
    ].copy()

    print("\nPRODUCT CATALOGUE")
    print("-" * 55)

    print("Total catalogue products:", len(catalog))

    print(
        "Products with missing descriptions:",
        len(missing_catalog)
    )

    print(
        "Products with available descriptions:",
        catalog["description"].notna().sum()
    )

    # Display missing product identifiers.
    print("\nMISSING DESCRIPTION PRODUCTS")
    print("-" * 55)

    print(
        missing_catalog[
            ["stock_code", "description"]
        ].to_string(index=False)
    )

    # Check how many product-day rows are affected.
    panel = pd.read_csv(
        PANEL_PATH,
        dtype={"stock_code": "string"},
        usecols=["stock_code"],
    )

    panel["stock_code"] = panel["stock_code"].str.strip()

    missing_codes = set(
        missing_catalog["stock_code"].dropna()
    )

    affected_rows = panel[
        panel["stock_code"].isin(missing_codes)
    ].shape[0]

    print("\nPRODUCT-DAY IMPACT")
    print("-" * 55)

    print("Affected product-day rows:", affected_rows)

    print(
        "Expected missing descriptions:",
        affected_rows
    )

    print("\nINVESTIGATION COMPLETED.")


if __name__ == "__main__":
    main()
