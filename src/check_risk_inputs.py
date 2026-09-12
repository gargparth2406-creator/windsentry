import pandas as pd


INPUT_FILE = (
    "data/processed/"
    "aeris_persistent_predictions.csv"
)


print("=" * 80)
print("AERIS RISK INPUT CHECK")
print("=" * 80)


# ==================================================
# 1. Load
# ==================================================

print("\nLoading persistent predictions...")

df = pd.read_csv(INPUT_FILE)

print("Shape:", df.shape)


# ==================================================
# 2. Required columns
# ==================================================

REQUIRED_COLUMNS = [
    "Turbine_ID",
    "Timestamp",
    "is_anomaly",
    "persistent_anomaly",
    "strong_persistent_anomaly",
    "power_deviation",
    "Gear_Bear_Temp_Avg"
]


print("\nRequired columns:")

for column in REQUIRED_COLUMNS:

    if column in df.columns:
        print(f"{column}: FOUND")
    else:
        print(f"{column}: MISSING")


# ==================================================
# 3. Missing values
# ==================================================

print("\nMissing values:")

print(
    df[REQUIRED_COLUMNS]
    .isna()
    .sum()
)


# ==================================================
# 4. Duplicate keys
# ==================================================

duplicates = df.duplicated(
    subset=["Turbine_ID", "Timestamp"],
    keep=False
)

print("\nDuplicate turbine-timestamp keys:")

print(
    df.loc[duplicates, ["Turbine_ID", "Timestamp"]]
    .drop_duplicates()
    .shape[0]
)


# ==================================================
# 5. Risk input statistics
# ==================================================

print("\nRisk input statistics:")

print(
    df[
        [
            "is_anomaly",
            "persistent_anomaly",
            "strong_persistent_anomaly",
            "power_deviation",
            "Gear_Bear_Temp_Avg"
        ]
    ].describe()
)


# ==================================================
# 6. Power deviation distribution
# ==================================================

print("\nPower deviation distribution:")

print(
    df["power_deviation"]
    .describe()
)


# ==================================================
# 7. Count positive power deviations
# ==================================================

positive_power = (
    df["power_deviation"] > 0
).sum()

negative_power = (
    df["power_deviation"] < 0
).sum()

zero_power = (
    df["power_deviation"] == 0
).sum()

print("\nPower deviation signs:")

print("Positive:", positive_power)
print("Negative:", negative_power)
print("Zero:", zero_power)


# ==================================================
# 8. Temperature thresholds
# ==================================================

temp = df["Gear_Bear_Temp_Avg"]

temp_95 = temp.quantile(0.95)
temp_99 = temp.quantile(0.99)

print("\nGearbox bearing temperature thresholds:")

print("95th percentile:", temp_95)
print("99th percentile:", temp_99)

print(
    ">= 95th percentile:",
    (temp >= temp_95).sum()
)

print(
    ">= 99th percentile:",
    (temp >= temp_99).sum()
)


# ==================================================
# 9. Anomaly counts
# ==================================================

print("\nAnomaly counts:")

print(
    df["is_anomaly"]
    .value_counts()
)


print("\nPersistent anomaly counts:")

print(
    df["persistent_anomaly"]
    .value_counts()
)


print("\nStrong persistent anomaly counts:")

print(
    df["strong_persistent_anomaly"]
    .value_counts()
)


# ==================================================
# 10. Check rows with anomalies
# ==================================================

print("\nAnomalous rows:")

print(
    df[
        df["is_anomaly"] == 1
    ][
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
)


# ==================================================
# 11. Check anomaly + power relationship
# ==================================================

print("\nAnomaly rows with power deviation:")

anomaly_df = df[
    df["is_anomaly"] == 1
]

print(
    anomaly_df[
        "power_deviation"
    ].describe()
)


print("\nAnomaly rows with positive power deviation:")

print(
    (
        anomaly_df["power_deviation"] > 0
    ).sum()
)


print("\nAnomaly rows with negative power deviation:")

print(
    (
        anomaly_df["power_deviation"] < 0
    ).sum()
)


# ==================================================
# 12. Final checks
# ==================================================

print("\nFinal checks:")

print(
    "Rows:",
    len(df)
)

print(
    "Unique keys:",
    df[
        ["Turbine_ID", "Timestamp"]
    ]
    .drop_duplicates()
    .shape[0]
)

print(
    "Duplicate rows:",
    duplicates.sum()
)

print(
    "Missing power deviation:",
    df["power_deviation"].isna().sum()
)

print(
    "Missing temperature:",
    df["Gear_Bear_Temp_Avg"].isna().sum()
)


print("\n" + "=" * 80)
print("RISK INPUT CHECK FINISHED")
print("=" * 80)