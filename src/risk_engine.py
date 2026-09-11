import pandas as pd
import numpy as np


INPUT_FILE = (
    "data/processed/"
    "aeris_persistent_predictions.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "aeris_risk_predictions.csv"
)


df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)


# ==================================================
# 1. Anomaly component
# ==================================================

anomaly_component = (
    df["is_anomaly"] * 30
)


# Persistent anomaly bonus
persistence_component = (
    df["persistent_anomaly"] * 15
)


strong_persistence_component = (
    df["strong_persistent_anomaly"] * 10
)


# ==================================================
# 2. Power degradation
# ==================================================

power_degradation = (
    df["power_deviation"]
    .clip(lower=0, upper=1)
)


power_component = (
    power_degradation * 30
)


# ==================================================
# 3. Gearbox temperature abnormality
# ==================================================

gear_temp = df[
    "Gear_Bear_Temp_Avg"
]

temp_95 = gear_temp.quantile(
    0.95
)

temp_99 = gear_temp.quantile(
    0.99
)


temperature_component = np.where(

    gear_temp >= temp_99,

    15,

    np.where(
        gear_temp >= temp_95,
        8,
        0
    )
)


# ==================================================
# 4. Total risk
# ==================================================

df["risk_score"] = (

    anomaly_component
    +
    persistence_component
    +
    strong_persistence_component
    +
    power_component
    +
    temperature_component

)


# Maximum = 100
df["risk_score"] = (
    df["risk_score"]
    .clip(0, 100)
)


# ==================================================
# 5. Risk classification
# ==================================================

def risk_category(score):

    if score >= 75:
        return "CRITICAL"

    elif score >= 50:
        return "HIGH"

    elif score >= 25:
        return "MEDIUM"

    else:
        return "LOW"


df["risk_level"] = (
    df["risk_score"]
    .apply(risk_category)
)


# ==================================================
# 6. Print summary
# ==================================================

print("\nRisk distribution:")

print(
    df["risk_level"]
    .value_counts()
)


print("\nHighest-risk records:")

print(
    df[
        [
            "Turbine_ID",
            "Timestamp",
            "risk_score",
            "risk_level"
        ]
    ]
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(20)
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)