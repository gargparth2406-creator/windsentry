import pandas as pd


INPUT_FILE = "data/processed/aeris_predictions.csv"


print("=" * 80)
print("AERIS MERGED PREDICTION VALIDATION")
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
# REQUIRED COLUMNS
# ============================================================

required = [
    "Turbine_ID",
    "Timestamp",
    "Grd_Prod_Pwr_Avg",
    "expected_power",
    "power_deviation",
    "anomaly_score",
    "is_anomaly"
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
# DATA TYPES
# ============================================================

print("\nData types:")

print(
    df[
        [
            c
            for c in required
            if c in df.columns
        ]
    ].dtypes
)


# ============================================================
# POWER STATISTICS
# ============================================================

print("\nPower statistics:")

power_columns = [
    "Grd_Prod_Pwr_Avg",
    "expected_power",
    "power_deviation"
]

for column in power_columns:

    if column in df.columns:

        print(
            f"\n{column}:"
        )

        print(
            df[column].describe()
        )


# ============================================================
# SAMPLE MISSING POWER ROWS
# ============================================================

if "power_deviation" in df.columns:

    missing = df[
        df["power_deviation"].isna()
    ]

    print(
        "\nRows with missing power_deviation:",
        len(missing)
    )

    if len(missing) > 0:

        print(
            missing[
                [
                    "Turbine_ID",
                    "Timestamp",
                    "Grd_Prod_Pwr_Avg",
                    "expected_power",
                    "power_deviation",
                    "anomaly_score",
                    "is_anomaly"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )


# ============================================================
# ACTUAL + EXPECTED BUT MISSING DEVIATION
# ============================================================

if all(
    c in df.columns
    for c in [
        "Grd_Prod_Pwr_Avg",
        "expected_power",
        "power_deviation"
    ]
):

    problem = (
        df["Grd_Prod_Pwr_Avg"].notna()
        &
        df["expected_power"].notna()
        &
        df["power_deviation"].isna()
    )

    print(
        "\nRows with valid actual + expected "
        "but missing deviation:",
        problem.sum()
    )


# ============================================================
# DUPLICATES
# ============================================================

if all(
    c in df.columns
    for c in [
        "Turbine_ID",
        "Timestamp"
    ]
):

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
# PER TURBINE
# ============================================================

if all(
    c in df.columns
    for c in [
        "Turbine_ID",
        "power_deviation"
    ]
):

    print(
        "\nMissing power deviation by turbine:"
    )

    print(
        df.groupby(
            "Turbine_ID"
        )["power_deviation"]
        .apply(
            lambda x: x.isna().sum()
        )
    )


print(
    "\n" + "=" * 80
)

print(
    "MERGED PREDICTION VALIDATION FINISHED"
)