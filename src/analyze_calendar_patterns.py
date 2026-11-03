import pandas as pd

DATA_PATH = "data/processed/daily_total_sales.csv"

df = pd.read_csv(DATA_PATH)

df["sales_date"] = pd.to_datetime(df["sales_date"])

df["weekday"] = df["sales_date"].dt.day_name()

df["is_zero_sales_day"] = df["total_units_sold"] == 0

summary = (
    df.groupby("weekday")
    .agg(
        total_days=("sales_date", "count"),
        zero_sales_days=("is_zero_sales_day", "sum"),
        average_units=("total_units_sold", "mean"),
    )
)

summary["zero_sales_percentage"] = (
    summary["zero_sales_days"] / summary["total_days"] * 100
)

print("\nWEEKDAY SALES ANALYSIS")
print(summary)

summary.to_csv(
    "reports/weekday_sales_analysis.csv"
)

print("\nAnalysis saved successfully.")      
# Investigate Saturdays with recorded sales

df["weekday_number"] = df["sales_date"].dt.day_of_week

saturday_sales = df[
    (df["weekday_number"] == 5) &
    (df["total_units_sold"] > 0)
]

print("\nSATURDAY SALES INVESTIGATION")
print("----------------------------")

print(saturday_sales[
    ["sales_date", "total_units_sold"]
].to_string(index=False))

print("\nNumber of Saturdays with sales:",
      len(saturday_sales))

print("\nTotal Saturday units:",
      saturday_sales["total_units_sold"].sum())