import pandas as pd
from pathlib import Path


INPUT_FILE = (
    "data/raw/"
    "Combined-Failure-Logbook-2016-2017.xlsx"
)

OUTPUT_FILE = (
    "data/processed/"
    "failure_log.csv"
)


print("=" * 80)
print("PROCESSING FAILURE LOG")
print("=" * 80)


df = pd.read_excel(INPUT_FILE)


# ------------------------------------------------
# Convert timestamp
# ------------------------------------------------

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)


# ------------------------------------------------
# Remove accidental whitespace
# ------------------------------------------------

df["Turbine_ID"] = (
    df["Turbine_ID"]
    .astype(str)
    .str.strip()
)

df["Component"] = (
    df["Component"]
    .astype(str)
    .str.strip()
)

df["Remarks"] = (
    df["Remarks"]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------
# Sort chronologically
# ------------------------------------------------

df = df.sort_values(
    ["Turbine_ID", "Timestamp"]
).reset_index(drop=True)


# ------------------------------------------------
# Turbines available in SCADA
# ------------------------------------------------

SCADA_TURBINES = [
    "T01",
    "T06",
    "T07",
    "T11"
]


df["scada_available"] = (
    df["Turbine_ID"]
    .isin(SCADA_TURBINES)
)


print("\nTotal failure records:", len(df))

print(
    "\nFailures with SCADA available:"
)

print(
    df["scada_available"]
    .value_counts()
)


print(
    "\nFailure records without matching SCADA:"
)

print(
    df[
        ~df["scada_available"]
    ][
        [
            "Turbine_ID",
            "Component",
            "Timestamp",
            "Remarks"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------
# Save
# ------------------------------------------------

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)