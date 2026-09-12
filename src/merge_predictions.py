import pandas as pd


POWER_FILE = "data/processed/power_predictions.csv"
ANOMALY_FILE = "data/processed/anomaly_predictions.csv"

OUTPUT_FILE = (
    "data/processed/"
    "aeris_predictions.csv"
)


KEY_COLUMNS = [
    "Turbine_ID",
    "Timestamp"
]


print("=" * 80)
print("AERIS PREDICTION MERGE")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

print("\nLoading power predictions...")

power = pd.read_csv(
    POWER_FILE
)

print("Power shape:", power.shape)


print("\nLoading anomaly predictions...")

anomaly = pd.read_csv(
    ANOMALY_FILE
)

print("Anomaly shape:", anomaly.shape)


# ============================================================
# TIMESTAMP NORMALIZATION
# ============================================================

power["Timestamp"] = pd.to_datetime(
    power["Timestamp"],
    utc=True
)

anomaly["Timestamp"] = pd.to_datetime(
    anomaly["Timestamp"],
    utc=True
)


# ============================================================
# DUPLICATE CHECK BEFORE MERGE
# ============================================================

print("\nDuplicate keys before merge:")

power_duplicates = power.duplicated(
    subset=KEY_COLUMNS
).sum()

anomaly_duplicates = anomaly.duplicated(
    subset=KEY_COLUMNS
).sum()

print(
    "Power duplicates:",
    power_duplicates
)

print(
    "Anomaly duplicates:",
    anomaly_duplicates
)


# ============================================================
# REMOVE SOURCE DUPLICATES
# ============================================================

print("\nRemoving duplicate turbine-timestamp records...")

power = (
    power
    .sort_values(KEY_COLUMNS)
    .drop_duplicates(
        subset=KEY_COLUMNS,
        keep="last"
    )
    .copy()
)

anomaly = (
    anomaly
    .sort_values(KEY_COLUMNS)
    .drop_duplicates(
        subset=KEY_COLUMNS,
        keep="last"
    )
    .copy()
)


print(
    "Power shape after deduplication:",
    power.shape
)

print(
    "Anomaly shape after deduplication:",
    anomaly.shape
)


# ============================================================
# KEEP USEFUL POWER COLUMNS
# ============================================================

power_columns = [
    "Turbine_ID",
    "Timestamp",
    "Grd_Prod_Pwr_Avg",
    "expected_power",
    "power_deviation"
]

power = power[
    [
        c
        for c in power_columns
        if c in power.columns
    ]
]


# ============================================================
# KEEP USEFUL ANOMALY COLUMNS
# ============================================================

anomaly_columns = [
    "Turbine_ID",
    "Timestamp",
    "anomaly_score",
    "is_anomaly",
    "Amb_WindSpeed_Avg",
    "Rtr_RPM_Avg",
    "Gen_RPM_Avg",
    "Gear_Oil_Temp_Avg",
    "Gear_Bear_Temp_Avg",
    "Gen_Bear_Temp_Avg",
    "Hyd_Oil_Temp_Avg",
    "Nac_Temp_Avg",
    "Blds_PitchAngle_Avg"
]

anomaly = anomaly[
    [
        c
        for c in anomaly_columns
        if c in anomaly.columns
    ]
]


# ============================================================
# MERGE
# ============================================================

print("\nMerging predictions...")

df = pd.merge(
    anomaly,
    power,
    on=KEY_COLUMNS,
    how="left",
    suffixes=("", "_power")
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    KEY_COLUMNS
).reset_index(
    drop=True
)


# ============================================================
# VALIDATION
# ============================================================

print("\nFinal shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


print("\nMissing values:")

print(
    df[
        [
            "expected_power",
            "power_deviation"
        ]
    ].isnull().sum()
)


print("\nDuplicate turbine-timestamp keys:")

final_duplicates = df.duplicated(
    subset=KEY_COLUMNS
).sum()

print(final_duplicates)


print("\nUnique turbines:")

print(
    df["Turbine_ID"]
    .nunique()
)


print("\nTurbines:")

print(
    sorted(
        df["Turbine_ID"]
        .dropna()
        .unique()
        .tolist()
    )
)


print("\nTimestamp range:")

print(
    df["Timestamp"].min(),
    "to",
    df["Timestamp"].max()
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 80)
print("AERIS PREDICTION MERGE FINISHED")
print("=" * 80)