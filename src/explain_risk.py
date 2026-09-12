import pandas as pd


INPUT_FILE = (
    "data/processed/"
    "aeris_risk_predictions.csv"
)


df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)


latest = (
    df.sort_values("Timestamp")
    .groupby("Turbine_ID")
    .tail(1)
    .copy()
)


def generate_explanation(row):

    reasons = []

    # --------------------------------
    # Anomaly
    # --------------------------------

    if row["is_anomaly"] == 1:

        reasons.append(
            "Unusual operating behavior detected"
        )


    # --------------------------------
    # Persistence
    # --------------------------------

    if row["persistent_anomaly"] == 1:

        reasons.append(
            "Anomaly is persistent"
        )


    # --------------------------------
    # Power degradation
    # --------------------------------

    if row["power_deviation"] > 0.15:

        reasons.append(
            "Significant power underperformance"
        )

    elif row["power_deviation"] > 0.05:

        reasons.append(
            "Moderate power underperformance"
        )


    # --------------------------------
    # Gearbox temperature
    # --------------------------------

    gear_temp = row[
        "Gear_Bear_Temp_Avg"
    ]

    if gear_temp > df[
        "Gear_Bear_Temp_Avg"
    ].quantile(0.99):

        reasons.append(
            "Gearbox bearing temperature unusually high"
        )


    if not reasons:

        reasons.append(
            "No major abnormal indicators detected"
        )


    return "; ".join(reasons)


latest["risk_explanation"] = (
    latest.apply(
        generate_explanation,
        axis=1
    )
)


print("\nAERIS EXPLANATIONS")
print("=" * 80)

for _, row in latest.iterrows():

    print(
        f"\n{row['Turbine_ID']}"
    )

    print(
        f"Risk: {row['risk_score']:.1f}"
    )

    print(
        f"Level: {row['risk_level']}"
    )

    print(
        f"Why: {row['risk_explanation']}"
    )