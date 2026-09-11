import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_FILE = "data/processed/edp_features.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Sort chronologically
df = df.sort_values(
    ["Turbine_ID", "Timestamp"]
).reset_index(drop=True)


# ---------------------------------------
# Features
# ---------------------------------------

FEATURES = [
    "Amb_WindSpeed_Avg",
    "Amb_WindSpeed_Std",
    "Rtr_RPM_Avg",
    "Rtr_RPM_Std",
    "Gen_RPM_Avg",
    "Blds_PitchAngle_Avg",
    "Amb_WindDir_Relative_Avg",
    "Amb_Temp_Avg"
]

TARGET = "Grd_Prod_Pwr_Avg"


# Remove rows with missing values
df = df.dropna(
    subset=FEATURES + [TARGET]
)


# ---------------------------------------
# Chronological split
# ---------------------------------------

split_time = df["Timestamp"].quantile(0.80)

train = df[
    df["Timestamp"] <= split_time
].copy()

test = df[
    df["Timestamp"] > split_time
].copy()


print("\nTraining rows:", len(train))
print("Testing rows:", len(test))

print("\nTraining until:", train["Timestamp"].max())
print("Testing from:", test["Timestamp"].min())


X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# ---------------------------------------
# Train model
# ---------------------------------------

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ---------------------------------------
# Predictions
# ---------------------------------------

predictions = model.predict(X_test)


# ---------------------------------------
# Evaluation
# ---------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\nMODEL RESULTS")
print("-------------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)


# ---------------------------------------
# Expected power
# ---------------------------------------

test["expected_power"] = predictions


# ---------------------------------------
# Power deviation
# ---------------------------------------

test["power_deviation"] = (
    test["expected_power"] -
    test[TARGET]
) / test["expected_power"].clip(lower=1)


# ---------------------------------------
# Save predictions
# ---------------------------------------

output_file = (
    "data/processed/"
    "power_predictions.csv"
)

test.to_csv(
    output_file,
    index=False
)


# ---------------------------------------
# Save model
# ---------------------------------------

model_file = (
    MODEL_DIR /
    "expected_power_model.joblib"
)

joblib.dump(
    model,
    model_file
)


print("\nSaved predictions:")
print(output_file)

print("\nSaved model:")
print(model_file)