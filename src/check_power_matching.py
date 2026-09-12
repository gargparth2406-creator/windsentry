import pandas as pd

DIAGNOSIS_FILE = "data/processed/component_diagnosis.csv"
POWER_FILE = "data/processed/aeris_predictions.csv"

print("Loading files...")

diagnosis = pd.read_csv(DIAGNOSIS_FILE)
power = pd.read_csv(POWER_FILE)

diagnosis["Timestamp"] = pd.to_datetime(
    diagnosis["Timestamp"],
    utc=True,
    errors="coerce"
)

power["Timestamp"] = pd.to_datetime(
    power["Timestamp"],
    utc=True,
    errors="coerce"
)

# Keep only the matching keys
diagnosis_keys = diagnosis[
    ["Turbine_ID", "Timestamp"]
].drop_duplicates()

power_keys = power[
    ["Turbine_ID", "Timestamp"]
].drop_duplicates()

print("\nDiagnosis unique keys:", len(diagnosis_keys))
print("Power unique keys:", len(power_keys))

# ------------------------------------------------------------
# Exact matching
# ------------------------------------------------------------

matched = diagnosis_keys.merge(
    power_keys,
    on=["Turbine_ID", "Timestamp"],
    how="inner"
)

print("\nExact matching keys:", len(matched))

# ------------------------------------------------------------
# Diagnosis keys missing from power
# ------------------------------------------------------------

missing = diagnosis_keys.merge(
    power_keys,
    on=["Turbine_ID", "Timestamp"],
    how="left",
    indicator=True
)

missing = missing[
    missing["_merge"] == "left_only"
].drop(columns=["_merge"])

print(
    "Diagnosis keys with NO exact power match:",
    len(missing)
)

print("\nFirst 20 missing keys:")
print(
    missing.head(20).to_string(index=False)
)

# ------------------------------------------------------------
# Compare turbine IDs
# ------------------------------------------------------------

print("\nDiagnosis turbines:")
print(
    sorted(
        diagnosis["Turbine_ID"].dropna().unique()
    )
)

print("\nPower turbines:")
print(
    sorted(
        power["Turbine_ID"].dropna().unique()
    )
)

# ------------------------------------------------------------
# Timestamp ranges
# ------------------------------------------------------------

print("\nDiagnosis timestamp range:")
print(
    diagnosis["Timestamp"].min(),
    "to",
    diagnosis["Timestamp"].max()
)

print("\nPower timestamp range:")
print(
    power["Timestamp"].min(),
    "to",
    power["Timestamp"].max()
)

# ------------------------------------------------------------
# Missing matches by turbine
# ------------------------------------------------------------

missing_by_turbine = (
    missing
    .groupby("Turbine_ID")
    .size()
    .sort_values(ascending=False)
)

print("\nMissing matches by turbine:")
print(missing_by_turbine)

# ------------------------------------------------------------
# Show whether timestamps exist but under another turbine
# ------------------------------------------------------------

diagnosis_timestamps = set(
    diagnosis["Timestamp"].dropna()
)

power_timestamps = set(
    power["Timestamp"].dropna()
)

common_timestamps = (
    diagnosis_timestamps
    & power_timestamps
)

print(
    "\nCommon timestamps ignoring Turbine_ID:",
    len(common_timestamps)
)

print(
    "Diagnosis unique timestamps:",
    len(diagnosis_timestamps)
)

print(
    "Power unique timestamps:",
    len(power_timestamps)
)

print("\nDone.")