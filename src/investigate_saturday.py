import pandas as pd

DATA_PATH = "data/processed/retail_demand_clean.csv"

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["invoice_date"],
    dtype={"invoice": "string"},
    low_memory=False
)

target_date = pd.Timestamp("2009-12-05")

saturday = df[
    df["invoice_date"].dt.normalize() == target_date
].copy()

print("\nSATURDAY TRANSACTION INVESTIGATION")
print("=" * 45)

print("Transaction rows:", len(saturday))

print(
    "Unique invoices:",
    saturday["invoice"].nunique()
)

print(
    "Total units:",
    saturday["quantity"].sum()
)

print(
    "First transaction:",
    saturday["invoice_date"].min()
)

print(
    "Last transaction:",
    saturday["invoice_date"].max()
)

print("\nTOP 10 PRODUCTS BY UNITS")

print(
    saturday.groupby("description")["quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nCOUNTRY DISTRIBUTION")

print(
    saturday["country"].value_counts().head(10)
)