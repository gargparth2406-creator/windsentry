import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_FILE = "data/processed/edp_features.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = "data/processed/power_predictions.csv"


print("=" * 80)
print("AERIS EXPECTED POWER PREDICTION")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

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


# ---------------------------------------
# Keep rows usable for model training
# ---------------------------------------

model_df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()


# ---------------------------------------
# Chronological split
# ---------------------------------------

split_time = model_df["Timestamp"].quantile(0.80)

train = model_df[
    model_df["Timestamp"] <= split_time
].copy()

test = model_df[
    model_df["Timestamp"] > split_time
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
# Evaluate on test data
# ---------------------------------------

print("\nEvaluating model...")

test_predictions = model.predict(X_test)


mae = mean_absolute_error(
    y_test,
    test_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

r2 = r2_score(
    y_test,
    test_predictions
)


print("\nMODEL RESULTS")
print("-------------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)


# ---------------------------------------
# Predict ALL usable rows
# ---------------------------------------

print("\nGenerating predictions for all usable rows...")

all_predictions = model.predict(
    model_df[FEATURES]
)

model_df["expected_power"] = all_predictions


# ---------------------------------------
# Power deviation
# ---------------------------------------

model_df["power_deviation"] = (
    model_df["expected_power"]
    - model_df[TARGET]
) / model_df["expected_power"].clip(lower=1)


# ---------------------------------------
# Save ALL usable predictions
# ---------------------------------------

print("\nSaving power predictions...")

model_df.to_csv(
    OUTPUT_FILE,
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
print(OUTPUT_FILE)

print("\nSaved model:")
print(model_file)

print("\nOutput shape:")
print(model_df.shape)

print("\nMissing power values:")
print(
    model_df[
        [
            "Grd_Prod_Pwr_Avg",
            "expected_power",
            "power_deviation"
        ]
    ].isna().sum()
)

print("\n" + "=" * 80)
print("EXPECTED POWER PREDICTION FINISHED")
print("=" * 80)