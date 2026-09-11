import pandas as pd
from pathlib import Path


POWER_FILE = "data/processed/power_predictions.csv"
ANOMALY_FILE = "data/processed/anomaly_predictions.csv"

OUTPUT_FILE = (
    "data/processed/"
    "aeris_predictions.csv"
)


print("Loading power predictions...")
power = pd.read_csv(POWER_FILE)

print("Loading anomaly predictions...")
anomaly = pd.read_csv(ANOMALY_FILE)


power["Timestamp"] = pd.to_datetime(
    power["Timestamp"]
)

anomaly["Timestamp"] = pd.to_datetime(
    anomaly["Timestamp"]
)


# Keep only the useful columns from power model
power_columns = [
    "Turbine_ID",
    "Timestamp",
    "Grd_Prod_Pwr_Avg",
    "expected_power",
    "power_deviation"
]

power = power[
    [c for c in power_columns if c in power.columns]
]


# Keep useful anomaly columns
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
    [c for c in anomaly_columns if c in anomaly.columns]
]


# Merge using the actual turbine and timestamp
df = pd.merge(
    anomaly,
    power,
    on=["Turbine_ID", "Timestamp"],
    how="left",
    suffixes=("", "_power")
)


# Sort
df = df.sort_values(
    ["Turbine_ID", "Timestamp"]
).reset_index(drop=True)


print("\nFinal shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(
    df[
        ["expected_power", "power_deviation"]
    ].isnull().sum()
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)