import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "data/processed/edp_features.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


print("Loading data...")

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)

df = df.sort_values(
    ["Turbine_ID", "Timestamp"]
).reset_index(drop=True)


# ---------------------------------------
# Features for anomaly detection
# ---------------------------------------

FEATURES = [

    "Amb_WindSpeed_Avg",

    "Rtr_RPM_Avg",

    "Gen_RPM_Avg",

    "Grd_Prod_Pwr_Avg",

    "Gear_Oil_Temp_Avg",

    "Gear_Bear_Temp_Avg",

    "Gen_Bear_Temp_Avg",

    "Hyd_Oil_Temp_Avg",

    "Nac_Temp_Avg",

    "Blds_PitchAngle_Avg"
]


df_model = df.dropna(
    subset=FEATURES
).copy()


X = df_model[FEATURES]


# ---------------------------------------
# Scale
# ---------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ---------------------------------------
# Isolation Forest
# ---------------------------------------

print("Training anomaly detection model...")

model = IsolationForest(

    n_estimators=200,

    contamination=0.02,

    random_state=42,

    n_jobs=-1

)


model.fit(X_scaled)


# ---------------------------------------
# Predictions
# ---------------------------------------

df_model["anomaly_prediction"] = model.predict(
    X_scaled
)

df_model["anomaly_score"] = -model.score_samples(
    X_scaled
)


# 1 = anomaly
# 0 = normal

df_model["is_anomaly"] = (
    df_model["anomaly_prediction"] == -1
).astype(int)


print("\nAnomaly counts:")

print(
    df_model["is_anomaly"].value_counts()
)


# ---------------------------------------
# Save results
# ---------------------------------------

output_file = (
    "data/processed/"
    "anomaly_predictions.csv"
)

df_model.to_csv(
    output_file,
    index=False
)


# ---------------------------------------
# Save models
# ---------------------------------------

joblib.dump(
    model,
    MODEL_DIR /
    "isolation_forest.joblib"
)

joblib.dump(
    scaler,
    MODEL_DIR /
    "anomaly_scaler.joblib"
)


print("\nSaved:")
print(output_file)

print(
    MODEL_DIR /
    "isolation_forest.joblib"
)