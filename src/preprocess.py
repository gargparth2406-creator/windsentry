import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "T01": "T01_F95.xlsx",
    "T06": "T06_F95.xlsx",
    "T07": "T07_F95.xlsx",
    "T11": "T11_F95.xlsx"
}


def load_turbine_data():

    all_data = []

    for turbine_id, filename in FILES.items():

        print(f"\nLoading {filename}...")

        path = RAW_DIR / filename

        df = pd.read_excel(path)

        # Add turbine identity
        df["turbine_id"] = turbine_id

        all_data.append(df)

        print(f"{turbine_id}: {df.shape}")


    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    return combined


def basic_cleaning(df):

    print("\nPerforming basic cleaning...")

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df


def main():

    df = load_turbine_data()

    print("\nCombined shape:")
    print(df.shape)

    df = basic_cleaning(df)

    print("\nAfter cleaning:")
    print(df.shape)

    df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
    )

    df = df.dropna(
    subset=["Timestamp"]
    )

    df = df.sort_values(
        ["turbine_id", "Timestamp"]
    )

    output_path = (
        PROCESSED_DIR /
        "edp_combined_raw.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved to: {output_path}"
    )


if __name__ == "__main__":
    main()