
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retail_transactions_combined.csv"
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def explore_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Combined dataset not found. "
            "Run build_dataset.py first."
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading combined dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["invoice_date"]
    )

    print("\nDATASET OVERVIEW")
    print("-" * 40)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print(
        "Date range:",
        df["invoice_date"].min(),
        "to",
        df["invoice_date"].max()
    )

    print(
        "Unique products:",
        df["stock_code"].nunique()
    )

    print(
        "Unique invoices:",
        df["invoice"].nunique()
    )

    # Create a positive-sales view for initial exploration.
    # This is NOT our final cleaned dataset.
    sales = df[
        (df["quantity"] > 0) &
        (df["unit_price"] > 0)
    ].copy()

    sales = sales.dropna(
        subset=["invoice_date"]
    )

    sales = sales[
        ~sales["invoice"]
        .astype(str)
        .str.startswith("C")
    ]

    print("\nPositive sales rows:", len(sales))

    # -------------------------------------------------
    # CHART 1: MONTHLY GROSS SALES
    # -------------------------------------------------

    sales["month"] = (
        sales["invoice_date"]
        .dt.to_period("M")
    )

    monthly_sales = (
        sales.groupby("month")["line_value"]
        .sum()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        monthly_sales.index.astype(str),
        monthly_sales.values,
        marker="o"
    )

    ax.set_title("Monthly Gross Sales")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales Value")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "monthly_gross_sales.png",
        dpi=150
    )

    plt.close(fig)

    # -------------------------------------------------
    # CHART 2: TOP 10 COUNTRIES BY INVOICES
    # -------------------------------------------------

    top_countries = (
        sales.groupby("country")["invoice"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    top_countries.plot(
        kind="barh",
        ax=ax
    )

    ax.set_title("Top 10 Countries by Invoice Count")
    ax.set_xlabel("Number of Invoices")
    ax.set_ylabel("Country")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "top_10_countries.png",
        dpi=150
    )

    plt.close(fig)

    # -------------------------------------------------
    # CHART 3: TOP 10 PRODUCTS BY UNITS SOLD
    # -------------------------------------------------

    product_sales = sales.dropna(
        subset=["description"]
    )

    top_products = (
        product_sales.groupby("description")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    top_products.plot(
        kind="barh",
        ax=ax
    )

    ax.set_title("Top 10 Products by Units Sold")
    ax.set_xlabel("Units Sold")
    ax.set_ylabel("Product")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "top_10_products.png",
        dpi=150
    )

    plt.close(fig)

    print("\nVISUALIZATIONS SAVED SUCCESSFULLY!")

    print("\nGenerated charts:")
    print("1. monthly_gross_sales.png")
    print("2. top_10_countries.png")
    print("3. top_10_products.png")

    print("\nAll charts saved in:", FIGURES_DIR)


if __name__ == "__main__":
    explore_data()
