import pandas as pd


INPUT_FILE = (
    "data/processed/"
    "aeris_predictions.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "aeris_persistent_predictions.csv"
)


df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)

df = df.sort_values(
    ["Turbine_ID", "Timestamp"]
).reset_index(drop=True)


# ---------------------------------------
# Count consecutive anomalous readings
# ---------------------------------------

groups = (
    df["is_anomaly"]
    .ne(
        df.groupby("Turbine_ID")["is_anomaly"]
        .shift()
    )
    .cumsum()
)

df["anomaly_streak"] = (
    df.groupby(
        ["Turbine_ID", groups]
    )
    .cumcount() + 1
)


# Only count streak when actually anomalous
df.loc[
    df["is_anomaly"] == 0,
    "anomaly_streak"
] = 0


# ---------------------------------------
# Persistence levels
# ---------------------------------------

df["persistent_anomaly"] = (
    df["anomaly_streak"] >= 3
).astype(int)


df["strong_persistent_anomaly"] = (
    df["anomaly_streak"] >= 6
).astype(int)


print("\nPersistence statistics:")

print(
    df[
        [
            "anomaly_streak",
            "persistent_anomaly",
            "strong_persistent_anomaly"
        ]
    ].describe()
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)