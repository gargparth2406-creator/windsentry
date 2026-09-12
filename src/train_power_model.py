import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# AERIS EXPECTED POWER PREDICTION
# ============================================================

INPUT_FILE = "data/processed/edp_features.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = "data/processed/power_predictions.csv"


print("=" * 80)
print("AERIS EXPECTED POWER PREDICTION")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    utc=True
)

df = df.sort_values(
    ["Turbine_ID", "Timestamp"]
).reset_index(drop=True)

print("Source shape:", df.shape)


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# TRAINING DATA
# ============================================================

print("\nPreparing training data...")

model_df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()

print(
    "Rows available for training:",
    len(model_df)
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

split_time = model_df["Timestamp"].quantile(0.80)

train = model_df[
    model_df["Timestamp"] <= split_time
].copy()

test = model_df[
    model_df["Timestamp"] > split_time
].copy()


print("\nTraining rows:", len(train))
print("Testing rows:", len(test))

print(
    "\nTraining until:",
    train["Timestamp"].max()
)

print(
    "Testing from:",
    test["Timestamp"].min()
)


X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# ============================================================
# TRAIN
# ============================================================

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


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating model...")

test_predictions = model.predict(
    X_test
)

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


# ============================================================
# GENERATE PREDICTIONS FOR ALL USABLE ROWS
# ============================================================

print(
    "\nGenerating predictions for all rows "
    "with valid power-model features..."
)

usable_mask = (
    df[FEATURES]
    .notna()
    .all(axis=1)
)

usable_rows = df.loc[
    usable_mask
].copy()

usable_predictions = model.predict(
    usable_rows[FEATURES]
)

usable_rows["expected_power"] = (
    usable_predictions
)

usable_rows["power_deviation"] = (
    usable_rows["expected_power"]
    - usable_rows[TARGET]
) / usable_rows["expected_power"].clip(
    lower=1
)


# ============================================================
# BUILD COMPLETE POWER OUTPUT
# ============================================================

print("\nBuilding complete prediction output...")

output = df.copy()

output["expected_power"] = np.nan
output["power_deviation"] = np.nan

output.loc[
    usable_mask,
    "expected_power"
] = usable_rows["expected_power"].values

output.loc[
    usable_mask,
    "power_deviation"
] = usable_rows["power_deviation"].values


# ============================================================
# SAVE
# ============================================================

print("\nSaving power predictions...")

output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODEL_DIR /
    "expected_power_model.joblib"
)

joblib.dump(
    model,
    model_file
)


# ============================================================
# VALIDATION
# ============================================================

print("\nSaved predictions:")
print(OUTPUT_FILE)

print("\nSaved model:")
print(model_file)

print("\nOutput shape:")
print(output.shape)

print("\nExpected-power missing values:")
print(
    output["expected_power"].isna().sum()
)

print("\nPower-deviation missing values:")
print(
    output["power_deviation"].isna().sum()
)

print("\nRequired columns:")
for column in [
    "Turbine_ID",
    "Timestamp",
    "Grd_Prod_Pwr_Avg",
    "expected_power",
    "power_deviation"
]:
    print(
        f"{column}:",
        "FOUND" if column in output.columns else "MISSING"
    )

print("\nDuplicate turbine-timestamp keys:")

print(
    output.duplicated(
        subset=[
            "Turbine_ID",
            "Timestamp"
        ]
    ).sum()
)

print("\n" + "=" * 80)
print("EXPECTED POWER PREDICTION FINISHED")
print("=" * 80)