import pandas as pd

POWER_FILE = "data/processed/aeris_predictions.csv"

df = pd.read_csv(POWER_FILE)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(
    df[
        [
            "Grd_Prod_Pwr_Avg",
            "expected_power",
            "power_deviation"
        ]
    ].isna().sum()
)

print("\nRows where expected_power exists:")
print(df["expected_power"].notna().sum())

print("\nRows where actual power exists:")
print(df["Grd_Prod_Pwr_Avg"].notna().sum())

print("\nRows where BOTH exist:")
print(
    (
        df["expected_power"].notna()
        & df["Grd_Prod_Pwr_Avg"].notna()
    ).sum()
)

print("\nExample rows with power:")
print(
    df[
        df["expected_power"].notna()
        & df["Grd_Prod_Pwr_Avg"].notna()
    ][
        [
            "Turbine_ID",
            "Timestamp",
            "Grd_Prod_Pwr_Avg",
            "expected_power",
            "power_deviation"
        ]
    ].head(10)
)

print("\nExample rows missing power:")
print(
    df[
        df["expected_power"].isna()
    ][
        [
            "Turbine_ID",
            "Timestamp",
            "Grd_Prod_Pwr_Avg",
            "expected_power",
            "power_deviation"
        ]
    ].head(10)
)