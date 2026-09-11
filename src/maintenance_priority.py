import pandas as pd


INPUT_FILE = (
    "data/processed/"
    "aeris_risk_predictions.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "maintenance_priority.csv"
)


df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)


# Latest available reading for each turbine
latest = (
    df.sort_values("Timestamp")
    .groupby("Turbine_ID")
    .tail(1)
    .copy()
)


# ---------------------------------------
# Priority score
# ---------------------------------------

latest["priority_score"] = (
    latest["risk_score"]
)


# ---------------------------------------
# Priority category
# ---------------------------------------

def priority(score):

    if score >= 75:
        return "P1 - IMMEDIATE INSPECTION"

    elif score >= 50:
        return "P2 - HIGH PRIORITY"

    elif score >= 25:
        return "P3 - MONITOR"

    else:
        return "P4 - NORMAL"


latest["maintenance_priority"] = (
    latest["priority_score"]
    .apply(priority)
)


latest = latest.sort_values(
    "priority_score",
    ascending=False
)


print("\nMAINTENANCE PRIORITY")
print("=" * 60)

print(
    latest[
        [
            "Turbine_ID",
            "risk_score",
            "risk_level",
            "maintenance_priority"
        ]
    ].to_string(index=False)
)


latest.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)