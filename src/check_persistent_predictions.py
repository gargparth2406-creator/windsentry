import pandas as pd


INPUT_FILE = (
    "data/processed/"
    "aeris_persistent_predictions.csv"
)


print("=" * 80)
print("AERIS PERSISTENT PREDICTION VALIDATION")
print("=" * 80)


df = pd.read_csv(INPUT_FILE)


# ============================================================
# BASIC
# ============================================================

print("\nShape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# REQUIRED RISK INPUTS
# ============================================================

required = [
    "Turbine_ID",
    "Timestamp",
    "is_anomaly",
    "persistent_anomaly",
    "strong_persistent_anomaly",
    "power_deviation",
    "Gear_Bear_Temp_Avg"
]


print("\nRequired columns:")

for column in required:

    print(
        f"{column}:",
        "FOUND" if column in df.columns else "MISSING"
    )


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

for column in required:

    if column in df.columns:

        print(
            f"{column}: "
            f"{df[column].isna().sum()}"
        )


# ============================================================
# DUPLICATES
# ============================================================

duplicates = df.duplicated(
    [
        "Turbine_ID",
        "Timestamp"
    ]
).sum()


print(
    "\nDuplicate turbine-timestamp keys:",
    duplicates
)


# ============================================================
# RISK INPUT VALIDATION
# ============================================================

risk_inputs = [
    "is_anomaly",
    "persistent_anomaly",
    "strong_persistent_anomaly",
    "power_deviation",
    "Gear_Bear_Temp_Avg"
]


available = [
    c
    for c in risk_inputs
    if c in df.columns
]


print("\nRisk input statistics:")

print(
    df[available].describe()
)


# ============================================================
# MISSING RISK INPUT ROWS
# ============================================================

if all(
    c in df.columns
    for c in risk_inputs
):

    missing_mask = (
        df[risk_inputs]
        .isna()
        .any(axis=1)
    )

    print(
        "\nRows with ANY missing risk input:",
        missing_mask.sum()
    )

    if missing_mask.sum() > 0:

        print(
            df.loc[
                missing_mask,
                [
                    "Turbine_ID",
                    "Timestamp",
                    "is_anomaly",
                    "persistent_anomaly",
                    "strong_persistent_anomaly",
                    "power_deviation",
                    "Gear_Bear_Temp_Avg"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )


# ============================================================
# POWER DEVIATION
# ============================================================

if "power_deviation" in df.columns:

    print(
        "\nPower deviation:"
    )

    print(
        df["power_deviation"].describe()
    )


# ============================================================
# PERSISTENCE
# ============================================================

if "persistent_anomaly" in df.columns:

    print(
        "\nPersistent anomaly distribution:"
    )

    print(
        df["persistent_anomaly"]
        .value_counts(dropna=False)
    )


if "strong_persistent_anomaly" in df.columns:

    print(
        "\nStrong persistent anomaly distribution:"
    )

    print(
        df["strong_persistent_anomaly"]
        .value_counts(dropna=False)
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "PERSISTENT PREDICTION VALIDATION FINISHED"
)