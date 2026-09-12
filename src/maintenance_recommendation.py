import pandas as pd
import numpy as np


# ============================================================
# AERIS MAINTENANCE RECOMMENDATION ENGINE
# ============================================================

INPUT_FILE = "data/processed/revenue_loss_predictions.csv"

OUTPUT_FILE = (
    "data/processed/maintenance_recommendations.csv"
)


# ============================================================
# COMPONENT PRIORITY
# ============================================================

COMPONENT_PRIORITY = {
    "GEARBOX": 5,
    "GENERATOR": 4,
    "GENERATOR_BEARING": 4,
    "HYDRAULIC_GROUP": 3,
    "TRANSFORMER": 3,
}


# ============================================================
# MAINTENANCE ACTIONS
# ============================================================

MAINTENANCE_ACTIONS = {

    "GEARBOX":
        "Inspect gearbox oil, bearings, lubrication system and gearbox temperature.",

    "GENERATOR":
        "Inspect generator bearings, generator temperature, RPM and electrical output.",

    "GENERATOR_BEARING":
        "Inspect generator bearing condition, temperature and lubrication.",

    "HYDRAULIC_GROUP":
        "Inspect hydraulic oil temperature, pitch system and hydraulic components.",

    "TRANSFORMER":
        "Inspect transformer phase temperatures, cooling system and electrical connections.",
}


# ============================================================
# FAILURE TIME ESTIMATION
# ============================================================

def estimate_failure_hours(row):

    risk = row.get("risk_score", 0)

    component_score = row.get(
        "component_score",
        0
    )

    strength = row.get(
        "diagnosis_strength",
        "LOW"
    )

    persistent = row.get(
        "persistent_anomaly",
        False
    )

    if pd.isna(risk):
        risk = 0

    if pd.isna(component_score):
        component_score = 0

    risk = float(risk)
    component_score = float(component_score)

    # --------------------------------------------------------
    # VERY HIGH RISK
    # --------------------------------------------------------

    if risk >= 80 or (
        strength == "STRONG"
        and component_score >= 3
    ):

        return 24

    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    if risk >= 60 or (
        strength == "STRONG"
        and component_score >= 2
    ):

        return 72

    # --------------------------------------------------------
    # MODERATE RISK
    # --------------------------------------------------------

    if risk >= 40 or (
        strength == "MODERATE"
        and component_score >= 2
    ):

        return 168

    # --------------------------------------------------------
    # WEAK / LOW RISK
    # --------------------------------------------------------

    if risk >= 20 or strength == "WEAK":

        return 336

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    return 720


# ============================================================
# PRIORITY
# ============================================================

def calculate_priority(row):

    risk = row.get("risk_score", 0)

    score = row.get(
        "component_score",
        0
    )

    component = row.get(
        "predicted_component",
        ""
    )

    strength = row.get(
        "diagnosis_strength",
        "LOW"
    )

    if pd.isna(risk):
        risk = 0

    if pd.isna(score):
        score = 0

    risk = float(risk)
    score = float(score)

    component_weight = COMPONENT_PRIORITY.get(
        component,
        1
    )

    priority_score = (
        risk * 0.50
        + min(score, 5) * 10 * 0.30
        + component_weight * 4 * 0.20
    )

    # --------------------------------------------------------
    # Priority classification
    # --------------------------------------------------------

    if (
        priority_score >= 70
        or strength == "STRONG"
    ):

        return "CRITICAL", priority_score

    if priority_score >= 50:

        return "HIGH", priority_score

    if priority_score >= 30:

        return "MEDIUM", priority_score

    return "LOW", priority_score


# ============================================================
# RECOMMENDATION
# ============================================================

def generate_recommendation(row):

    component = row.get(
        "predicted_component",
        "UNKNOWN"
    )

    priority = row.get(
        "maintenance_priority",
        "LOW"
    )

    action = MAINTENANCE_ACTIONS.get(
        component,
        "Perform general turbine inspection."
    )

    if priority == "CRITICAL":

        return (
            "Immediate inspection required. "
            + action
        )

    if priority == "HIGH":

        return (
            "Schedule maintenance as soon as possible. "
            + action
        )

    if priority == "MEDIUM":

        return (
            "Schedule inspection during the next maintenance window. "
            + action
        )

    return (
        "Continue monitoring. "
        + action
    )


# ============================================================
# FAILURE WINDOW
# ============================================================

def failure_window(hours):

    if hours <= 24:

        return "Within 24 hours"

    if hours <= 72:

        return "Within 3 days"

    if hours <= 168:

        return "Within 7 days"

    if hours <= 336:

        return "Within 14 days"

    return "More than 30 days"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "AERIS MAINTENANCE RECOMMENDATION ENGINE"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        "\nLoading revenue-loss predictions..."
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Timestamp"]
    )

    print(
        f"Input shape: {df.shape}"
    )

    # --------------------------------------------------------
    # FAILURE TIME
    # --------------------------------------------------------

    print(
        "\nEstimating component failure windows..."
    )

    df["estimated_failure_hours"] = (
        df.apply(
            estimate_failure_hours,
            axis=1
        )
    )

    df["estimated_failure_time"] = (
        df["Timestamp"]
        + pd.to_timedelta(
            df["estimated_failure_hours"],
            unit="h"
        )
    )

    df["failure_window"] = (
        df["estimated_failure_hours"]
        .apply(failure_window)
    )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    print(
        "Calculating maintenance priorities..."
    )

    priority_results = df.apply(
        calculate_priority,
        axis=1
    )

    df["maintenance_priority"] = (
        priority_results
        .apply(lambda x: x[0])
    )

    df["maintenance_priority_score"] = (
        priority_results
        .apply(lambda x: x[1])
    )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    print(
        "Generating maintenance recommendations..."
    )

    df["maintenance_recommendation"] = (
        df.apply(
            generate_recommendation,
            axis=1
        )
    )

    # --------------------------------------------------------
    # REASON
    # --------------------------------------------------------

    df["maintenance_reason"] = (
        df.apply(
            lambda row:
            (
                f"{row['predicted_component']} identified as "
                f"the most likely affected component with "
                f"a component score of "
                f"{row['component_score']:.2f} and "
                f"risk level {row['risk_level']}."
            ),
            axis=1
        )
    )

    # --------------------------------------------------------
    # OUTPUT COLUMNS
    # --------------------------------------------------------

    output_columns = [

        "Turbine_ID",
        "Timestamp",

        "risk_score",
        "risk_level",

        "predicted_component",
        "component_score",
        "diagnosis_strength",

        "maintenance_priority",
        "maintenance_priority_score",

        "estimated_failure_hours",
        "estimated_failure_time",
        "failure_window",

        "maintenance_recommendation",
        "maintenance_reason",

        "power_loss_kw",
        "estimated_24h_revenue_loss",
        "estimated_7d_revenue_loss",

        "risk_weighted_revenue_exposure",
    ]

    output_columns = [
        c
        for c in output_columns
        if c in df.columns
    ]

    result = df[
        output_columns
    ].copy()

    # --------------------------------------------------------
    # ROUND NUMBERS
    # --------------------------------------------------------

    numeric_columns = (
        result
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    result[numeric_columns] = (
        result[numeric_columns]
        .round(3)
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\nMaintenance recommendation complete."
    )

    print(
        f"Output shape: {result.shape}"
    )

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )

    print(
        "\nMaintenance priority distribution:"
    )

    print(
        result[
            "maintenance_priority"
        ].value_counts()
    )

    print(
        "\nMost at-risk components:"
    )

    print(
        result[
            "predicted_component"
        ].value_counts()
    )

    print(
        "\nTop maintenance cases:"
    )

    top = result.sort_values(
        [
            "maintenance_priority_score",
            "estimated_24h_revenue_loss"
        ],
        ascending=False
    ).head(10)

    print(
        top[
            [
                "Turbine_ID",
                "Timestamp",
                "risk_level",
                "predicted_component",
                "maintenance_priority",
                "failure_window",
                "estimated_24h_revenue_loss",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "MAINTENANCE RECOMMENDATION ENGINE FINISHED"
    )


if __name__ == "__main__":
    main()