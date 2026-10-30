#DemandIQ - Day1 
# First Retail Dataset Inspection
import pandas as pd 
from pathlib import Path

#locate file 
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
#dataset location 
DATA_PATH = PROJECT_ROOT/"data"/"raw"/"online_retail_II.xlsx"
#check if file exists 
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )
# 4. Read the first 10,000 rows
df = pd.read_excel(
    DATA_PATH,
    sheet_name=0,
    nrows=10000
)

# 5. Display basic information
print("\n========== DEMANDIQ DATA INSPECTION ==========")

print("\n1. First 5 rows:")
print(df.head())

print("\n2. Dataset shape:")
print(df.shape)

print("\n3. Column names:")
print(df.columns.tolist())

print("\n4. Data types:")
print(df.dtypes)

print("\n5. Missing values:")
print(df.isnull().sum())

print("\n6. Statistical summary:")
print(df.describe(include="all"))

print("\n========== INSPECTION COMPLETE ==========")