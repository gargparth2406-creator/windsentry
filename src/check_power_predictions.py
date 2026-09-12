import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/power_predictions.csv"


print("=" * 80)
print("AERIS POWER PREDICTION VALIDATION")
print("=" * 80)


df = pd.read_csv(INPUT_FILE)


# ------------------------------------------------------------
# BASIC INFORMATION
# ------------------------------------------------------------

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# REQUIRED COLUMNS
# ------------------------------------------------------------

required = [
    "Turbine_ID",
    "Timestamp",
    "Grd_Prod_Pwr_Avg",
    "expected_power",
    "power_deviation"
]

print("\nRequired columns:")

for column in required:

    print(
        f"{column}:",
        "FOUND" if column in df.columns else "MISSING"
    )


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values:")

for column in required:

    if column in df.columns:

        print(
            f"{column}: "
            f"{df[column].isna().sum()}"
        )


# ------------------------------------------------------------
# NUMERIC CONVERSION
# ------------------------------------------------------------

numeric_columns = [
    "Grd_Prod_Pwr_Avg",
    "expected_power",
    "power_deviation"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ------------------------------------------------------------
# VALID POWER DEVIATION
# ------------------------------------------------------------

print("\nPower deviation statistics:")

if "power_deviation" in df.columns:

    print(
        df["power_deviation"].describe()
    )


# ------------------------------------------------------------
# SAMPLE MISSING RECORDS
# ------------------------------------------------------------

print("\nSample rows with missing power_deviation:")

missing = df[
    df["power_deviation"].isna()
]

if len(missing) > 0:

    print(
        missing[
            [
                "Turbine_ID",
                "Timestamp",
                "Grd_Prod_Pwr_Avg",
                "expected_power",
                "power_deviation"
            ]
        ].head(20).to_string(index=False)
    )

else:

    print("NONE")


# ------------------------------------------------------------
# CHECK EXPECTED POWER
# ------------------------------------------------------------

print("\nExpected power statistics:")

print(
    df["expected_power"].describe()
)


# ------------------------------------------------------------
# CHECK ACTUAL POWER
# ------------------------------------------------------------

print("\nActual power statistics:")

print(
    df["Grd_Prod_Pwr_Avg"].describe()
)


# ------------------------------------------------------------
# RECONSTRUCT DEVIATION
# ------------------------------------------------------------

print("\nReconstructing power deviation...")

valid = (
    df["Grd_Prod_Pwr_Avg"].notna()
    &
    df["expected_power"].notna()
)

reconstructed = (
    (
        df.loc[valid, "expected_power"]
        -
        df.loc[valid, "Grd_Prod_Pwr_Avg"]
    )
    /
    df.loc[valid, "expected_power"]
)


print("\nRows with valid actual + expected power:")
print(valid.sum())


print("\nRows with missing power_deviation but valid inputs:")

problem = (
    df["power_deviation"].isna()
    &
    valid
)

print(problem.sum())


# ------------------------------------------------------------
# COMPARISON
# ------------------------------------------------------------

if problem.sum() > 0:

    print(
        "\nThis indicates power_deviation is missing "
        "even though actual and expected power exist."
    )

    print(
        df.loc[
            problem,
            [
                "Turbine_ID",
                "Timestamp",
                "Grd_Prod_Pwr_Avg",
                "expected_power",
                "power_deviation"
            ]
        ].head(20).to_string(index=False)
    )


# ------------------------------------------------------------
# PER TURBINE
# ------------------------------------------------------------

print("\nMissing power_deviation by turbine:")

print(
    df.groupby("Turbine_ID")[
        "power_deviation"
    ]
    .apply(lambda x: x.isna().sum())
)


print("\n" + "=" * 80)
print("POWER PREDICTION VALIDATION FINISHED")
print("=" * 80)