import pandas as pd
import numpy as np


INPUT_FILE = (
    "data/processed/"
    "aeris_persistent_predictions.csv"
)


print("=" * 80)
print("AERIS RISK COMPONENT CHECK")
print("=" * 80)


# ==================================================
# 1. Load
# ==================================================

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)


# ==================================================
# 2. Calculate components exactly as risk_engine.py
# ==================================================

# Anomaly
df["anomaly_component"] = (
    df["is_anomaly"] * 30
)


# Persistence
df["persistence_component"] = (
    df["persistent_anomaly"] * 15
)


# Strong persistence
df["strong_persistence_component"] = (
    df["strong_persistent_anomaly"] * 10
)


# Power degradation
df["power_degradation"] = (
    df["power_deviation"]
    .clip(lower=0, upper=1)
)

df["power_component"] = (
    df["power_degradation"] * 30
)


# Temperature
gear_temp = df["Gear_Bear_Temp_Avg"]

temp_95 = gear_temp.quantile(0.95)
temp_99 = gear_temp.quantile(0.99)


df["temperature_component"] = np.where(
    gear_temp >= temp_99,
    15,
    np.where(
        gear_temp >= temp_95,
        8,
        0
    )
)


# ==================================================
# 3. Total score
# ==================================================

df["risk_score"] = (
    df["anomaly_component"]
    +
    df["persistence_component"]
    +
    df["strong_persistence_component"]
    +
    df["power_component"]
    +
    df["temperature_component"]
)


df["risk_score"] = (
    df["risk_score"]
    .clip(0, 100)
)


# ==================================================
# 4. Component statistics
# ==================================================

COMPONENTS = [
    "anomaly_component",
    "persistence_component",
    "strong_persistence_component",
    "power_component",
    "temperature_component",
    "risk_score"
]


print("\nComponent statistics:")

print(
    df[COMPONENTS]
    .describe()
)


# ==================================================
# 5. How often each component contributes
# ==================================================

print("\nComponent activation counts:")

print(
    "Anomaly component > 0:",
    (df["anomaly_component"] > 0).sum()
)

print(
    "Persistence component > 0:",
    (df["persistence_component"] > 0).sum()
)

print(
    "Strong persistence component > 0:",
    (df["strong_persistence_component"] > 0).sum()
)

print(
    "Power component > 0:",
    (df["power_component"] > 0).sum()
)

print(
    "Temperature component > 0:",
    (df["temperature_component"] > 0).sum()
)


# ==================================================
# 6. Highest-risk records with components
# ==================================================

print("\nHighest-risk records with components:")

highest = (
    df[
        [
            "Turbine_ID",
            "Timestamp",
            "is_anomaly",
            "persistent_anomaly",
            "strong_persistent_anomaly",
            "power_deviation",
            "Gear_Bear_Temp_Avg",
            "anomaly_component",
            "persistence_component",
            "strong_persistence_component",
            "power_component",
            "temperature_component",
            "risk_score"
        ]
    ]
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(30)
)


print(highest.to_string(index=False))


# ==================================================
# 7. Records capable of becoming CRITICAL
# ==================================================

critical_candidates = df[
    df["risk_score"] >= 70
].copy()


print("\nRecords with risk >= 70:")

print(
    len(critical_candidates)
)


print("\nRisk >= 70 component summary:")

print(
    critical_candidates[
        [
            "anomaly_component",
            "persistence_component",
            "strong_persistence_component",
            "power_component",
            "temperature_component",
            "risk_score"
        ]
    ].describe()
)


# ==================================================
# 8. Maximum possible score by current state
# ==================================================

print("\nMaximum observed component values:")

for column in [
    "anomaly_component",
    "persistence_component",
    "strong_persistence_component",
    "power_component",
    "temperature_component"
]:
    print(
        column,
        ":",
        df[column].max()
    )


print(
    "\nMaximum observed risk score:",
    df["risk_score"].max()
)


# ==================================================
# 9. Risk score distribution
# ==================================================

print("\nRisk score quantiles:")

print(
    df["risk_score"].quantile(
        [
            0.50,
            0.90,
            0.95,
            0.99,
            0.995,
            0.999,
            1.00
        ]
    )
)


print("\n" + "=" * 80)
print("RISK COMPONENT CHECK FINISHED")
print("=" * 80)