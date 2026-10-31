
from pathlib import Path
import pandas as pd


# Find the main DemandIQ folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Location of the original Excel workbook
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"


def inspect_workbook():

    # Check whether the dataset exists
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    print("=" * 60)
    print("DEMANDIQ - DATASET INSPECTION")
    print("=" * 60)

    # Read Excel workbook information
    excel = pd.ExcelFile(DATA_FILE)

    print("\nWORKSHEETS:")
    print(excel.sheet_names)

    # Inspect each worksheet separately
    for sheet in excel.sheet_names:

        print("\n" + "=" * 60)
        print(f"INSPECTING SHEET: {sheet}")
        print("=" * 60)

        # Read only the first 5,000 rows for initial inspection
        df = pd.read_excel(
            DATA_FILE,
            sheet_name=sheet,
            nrows=5000
        )

        print("\nFIRST 5 ROWS:")
        print(df.head().to_string(index=False))

        print("\nDATASET DIMENSIONS:")
        print(df.shape)

        print("\nCOLUMN NAMES:")
        print(df.columns.tolist())

        print("\nDATA TYPES:")
        print(df.dtypes.to_string())

        print("\nMISSING VALUES:")
        print(df.isnull().sum().to_string())

        print("\nUNIQUE VALUES:")
        print(df.nunique().to_string())

        print("\nDUPLICATE ROWS IN SAMPLE:")
        print(df.duplicated().sum())

    print("\nINSPECTION COMPLETED!")


if __name__ == "__main__":
    inspect_workbook()
