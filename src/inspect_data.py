import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

files = [
    "T01_F95.xlsx",
    "T06_F95.xlsx",
    "T07_F95.xlsx",
    "T11_F95.xlsx"
]

for file in files:

    path = DATA_DIR / file

    print("\n" + "=" * 80)
    print(f"FILE: {file}")
    print("=" * 80)

    df = pd.read_excel(path)

    print("Shape:", df.shape)

    print("\nColumns:")
    for i, col in enumerate(df.columns):
        print(f"{i:02d} : {col}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\n")