import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/anomaly_predictions.csv"

OUTPUT_FILE = (
    "data/processed/"
    "turbine_health_scores.csv"
)


df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)


# ---------------------------------------
# Power deviation
# ---------------------------------------

# If expected_power is available
if "expected_power" in df.columns:

    df["power_deviation"] = (
        df["expected_power"] -
        df["Grd_Prod_Pwr_Avg"]
    ) / df["expected_power"].clip(lower=1)

else:

    df["power_deviation"] = 0


# ---------------------------------------
# Normalize power degradation
# ---------------------------------------

df["power_risk"] = (
    df["power_deviation"]
    .clip(lower=0)
    .clip(upper=1)
)


# ---------------------------------------
# Temperature risk
# ---------------------------------------

bearing_temp = df[
    "Gear_Bear_Temp_Avg"
]

temp_95 = bearing_temp.quantile(
    0.95
)

temp_risk = (
    (bearing_temp - temp_95) /
    (temp_95 * 0.20)
)

df["temperature_risk"] = (
    temp_risk
    .clip(lower=0)
    .clip(upper=1)
)


# ---------------------------------------
# Anomaly risk
# ---------------------------------------

df["anomaly_risk"] = (
    df["is_anomaly"]
)


# ---------------------------------------
# Combined health risk
# ---------------------------------------

df["risk_score"] = (

    0.45 *
    df["anomaly_risk"]

    +

    0.35 *
    df["power_risk"]

    +

    0.20 *
    df["temperature_risk"]

)


# ---------------------------------------
# Convert to 0-100
# ---------------------------------------

df["risk_score"] = (
    df["risk_score"] * 100
)


# ---------------------------------------
# Risk category
# ---------------------------------------

def classify_risk(score):

    if score >= 70:
        return "CRITICAL"

    elif score >= 40:
        return "WARNING"

    elif score >= 20:
        return "WATCH"

    else:
        return "NORMAL"


df["risk_level"] = (
    df["risk_score"]
    .apply(classify_risk)
)


print(
    df[
        [
            "Turbine_ID",
            "Timestamp",
            "risk_score",
            "risk_level"
        ]
    ].tail()
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:", OUTPUT_FILE)