import pandas as pd


# ============================================================
# AERIS SOURCE DATA DUPLICATE CHECK
# ============================================================

INPUT_FILE = "data/processed/edp_features.csv"

KEY_COLUMNS = [
    "Turbine_ID",
    "Timestamp"
]


print("=" * 80)
print("AERIS SOURCE DATA DUPLICATE CHECK")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

print("\nLoading source dataset...")

df = pd.read_csv(
    INPUT_FILE
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    utc=True,
    errors="coerce"
)

print(
    f"Total rows: {len(df)}"
)


# ============================================================
# BASIC CHECK
# ============================================================

print("\nChecking required columns...")

for column in KEY_COLUMNS:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )

print("Required columns: FOUND")


# ============================================================
# MISSING KEYS
# ============================================================

print("\nMissing key values:")

print(
    df[KEY_COLUMNS]
    .isna()
    .sum()
)


# ============================================================
# UNIQUE KEYS
# ============================================================

unique_keys = (
    df[KEY_COLUMNS]
    .drop_duplicates()
)

print(
    "\nUnique Turbine_ID + Timestamp keys:",
    len(unique_keys)
)


# ============================================================
# DUPLICATE KEYS
# ============================================================

duplicate_mask = df.duplicated(
    subset=KEY_COLUMNS,
    keep=False
)

duplicate_rows = df[
    duplicate_mask
].copy()

duplicate_keys = (
    duplicate_rows[KEY_COLUMNS]
    .drop_duplicates()
)

print(
    "\nDuplicate turbine-timestamp keys:",
    len(duplicate_keys)
)

print(
    "Rows belonging to duplicate keys:",
    len(duplicate_rows)
)


# ============================================================
# DUPLICATES BY TURBINE
# ============================================================

print("\nDuplicate keys by turbine:")

if len(duplicate_keys) > 0:

    print(
        duplicate_keys[
            "Turbine_ID"
        ]
        .value_counts()
        .sort_index()
    )

else:

    print("NONE")


# ============================================================
# SAMPLE DUPLICATE KEYS
# ============================================================

print("\nSample duplicate keys:")

if len(duplicate_keys) > 0:

    print(
        duplicate_keys
        .sort_values(KEY_COLUMNS)
        .head(20)
        .to_string(index=False)
    )

else:

    print("NONE")


# ============================================================
# SHOW ACTUAL DUPLICATE RECORDS
# ============================================================

if len(duplicate_rows) > 0:

    print("\nSample duplicate records:")

    sample_keys = (
        duplicate_keys
        .sort_values(KEY_COLUMNS)
        .head(5)
    )

    sample_records = pd.merge(
        sample_keys,
        df,
        on=KEY_COLUMNS,
        how="left"
    )

    print(
        sample_records[
            [
                "Turbine_ID",
                "Timestamp",
                "Grd_Prod_Pwr_Avg",
                "Amb_WindSpeed_Avg",
                "Rtr_RPM_Avg",
                "Gen_RPM_Avg"
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# PER-TURBINE TOTALS
# ============================================================

print("\nRows and unique keys by turbine:")

turbine_summary = (
    df.groupby("Turbine_ID")
    .agg(
        total_rows=("Turbine_ID", "size"),
        unique_timestamps=("Timestamp", "nunique")
    )
    .reset_index()
)

turbine_summary["duplicate_rows"] = (
    turbine_summary["total_rows"]
    - turbine_summary["unique_timestamps"]
)

print(
    turbine_summary
    .to_string(index=False)
)


# ============================================================
# FINAL CONSISTENCY CHECK
# ============================================================

print("\n" + "=" * 80)
print("SOURCE DATA CHECK RESULT")
print("=" * 80)

if len(duplicate_keys) == 0:

    print(
        "\nPASS: No duplicate Turbine_ID + Timestamp keys found."
    )

else:

    print(
        "\nWARNING: Duplicate Turbine_ID + Timestamp keys found."
    )

    print(
        "These duplicates must be investigated before fixing the merge."
    )


if df[KEY_COLUMNS].isna().any().any():

    print(
        "WARNING: Missing values exist in the key columns."
    )

else:

    print(
        "PASS: No missing key values."
    )


print(
    "\nTotal rows:",
    len(df)
)

print(
    "Unique keys:",
    len(unique_keys)
)

print(
    "Duplicate keys:",
    len(duplicate_keys)
)

print(
    "\n" + "=" * 80
)
print("SOURCE DATA DUPLICATE CHECK FINISHED")
print("=" * 80)