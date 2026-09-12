import pandas as pd
import numpy as np


# ============================================================
# AERIS COMPONENT DIAGNOSIS
# Fast vectorized implementation
# ============================================================

COMPONENT_FEATURES = {
    "GEARBOX": [
        "Gear_Oil_Temp_Avg",
        "Gear_Bear_Temp_Avg",
        "Gen_RPM_Avg",
        "Rtr_RPM_Avg",
        "power_deviation",
        "anomaly_score",
    ],

    "GENERATOR": [
        "Gen_Bear_Temp_Avg",
        "Gen_RPM_Avg",
        "Grd_Prod_Pwr_Avg",
        "power_deviation",
        "anomaly_score",
    ],

    "GENERATOR_BEARING": [
        "Gen_Bear_Temp_Avg",
        "Gen_RPM_Avg",
        "power_deviation",
        "anomaly_score",
    ],

    "HYDRAULIC_GROUP": [
        "Hyd_Oil_Temp_Avg",
        "Blds_PitchAngle_Avg",
        "Rtr_RPM_Avg",
        "power_deviation",
        "anomaly_score",
    ],

    "TRANSFORMER": [
        "HVTrafo_Phase1_Temp_Avg",
        "HVTrafo_Phase2_Temp_Avg",
        "HVTrafo_Phase3_Temp_Avg",
        "Grd_Prod_Pwr_Avg",
        "power_deviation",
        "anomaly_score",
    ],
}


# ============================================================
# BUILD TURBINE BASELINES
# ============================================================

def build_baselines(df):

    sensor_columns = [
        "Gear_Oil_Temp_Avg",
        "Gear_Bear_Temp_Avg",
        "Gen_Bear_Temp_Avg",
        "Gen_RPM_Avg",
        "Rtr_RPM_Avg",
        "Hyd_Oil_Temp_Avg",
        "Blds_PitchAngle_Avg",
        "Grd_Prod_Pwr_Avg",
        "HVTrafo_Phase1_Temp_Avg",
        "HVTrafo_Phase2_Temp_Avg",
        "HVTrafo_Phase3_Temp_Avg",
        "power_deviation",
        "anomaly_score",
    ]

    available = [
        c for c in sensor_columns
        if c in df.columns
    ]

    baselines = (
        df.groupby("Turbine_ID")[available]
        .agg(["median", "std"])
    )

    return baselines


# ============================================================
# FAST ABNORMALITY CALCULATION
# ============================================================

def abnormality_series(df, baselines, feature):

    if feature not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index
        )

    if feature not in baselines.columns.get_level_values(0):
        return pd.Series(
            np.nan,
            index=df.index
        )

    medians = df["Turbine_ID"].map(
        baselines[feature]["median"]
    )

    stds = df["Turbine_ID"].map(
        baselines[feature]["std"]
    )

    values = df[feature]

    result = (values - medians).abs()

    valid_std = (
        stds.notna()
        & (stds > 0)
    )

    result.loc[valid_std] = (
        result.loc[valid_std]
        / stds.loc[valid_std]
    )

    result.loc[
        values.isna() | medians.isna()
    ] = np.nan

    return result


# ============================================================
# COMPONENT SCORE
# ============================================================

def calculate_component_score(
    df,
    baselines,
    component
):

    features = COMPONENT_FEATURES[component]

    abnormalities = []

    for feature in features:

        abnormality = abnormality_series(
            df,
            baselines,
            feature
        )

        abnormality = abnormality.clip(
            lower=0,
            upper=5
        )

        abnormalities.append(abnormality)

    if not abnormalities:
        return pd.Series(
            0.0,
            index=df.index
        )

    values = pd.concat(
        abnormalities,
        axis=1
    )

    return values.mean(
        axis=1,
        skipna=True
    ).fillna(0.0)


# ============================================================
# DIAGNOSIS STRENGTH
# ============================================================

def calculate_strength(
    score,
    persistent
):

    conditions = [
        (score >= 3) & persistent,
        score >= 2,
        score >= 1
    ]

    choices = [
        "STRONG",
        "MODERATE",
        "WEAK"
    ]

    return np.select(
        conditions,
        choices,
        default="LOW"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AERIS COMPONENT DIAGNOSIS ENGINE")
    print("=" * 80)

    input_file = (
        "data/processed/aeris_risk_predictions.csv"
    )

    output_file = (
        "data/processed/component_diagnosis.csv"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\nLoading AERIS predictions...")

    df = pd.read_csv(
        input_file,
        parse_dates=["Timestamp"]
    )

    print(
        f"Input shape: {df.shape}"
    )

    # --------------------------------------------------------
    # BASELINES
    # --------------------------------------------------------

    print(
        "\nBuilding turbine-specific baselines..."
    )

    baselines = build_baselines(df)

    print("Baselines created.")

    # --------------------------------------------------------
    # COMPONENT SCORES
    # --------------------------------------------------------

    print(
        "\nCalculating component scores..."
    )

    component_scores = {}

    for component in COMPONENT_FEATURES:

        print(
            f"  → {component}"
        )

        component_scores[component] = (
            calculate_component_score(
                df,
                baselines,
                component
            )
        )

    # --------------------------------------------------------
    # CREATE SCORE TABLE
    # --------------------------------------------------------

    score_df = pd.DataFrame(
        component_scores,
        index=df.index
    )

    # --------------------------------------------------------
    # FIND TOP COMPONENT
    # --------------------------------------------------------

    print(
        "\nDetermining most likely failed component..."
    )

    df["predicted_component"] = (
        score_df.idxmax(axis=1)
    )

    df["component_score"] = (
        score_df.max(axis=1).round(3)
    )

    # --------------------------------------------------------
    # DIAGNOSIS STRENGTH
    # --------------------------------------------------------

    persistent = (
        df["persistent_anomaly"]
        .fillna(False)
        .astype(bool)
        if "persistent_anomaly" in df.columns
        else pd.Series(
            False,
            index=df.index
        )
    )

    df["diagnosis_strength"] = (
        calculate_strength(
            df["component_score"],
            persistent
        )
    )

    # --------------------------------------------------------
    # ADD INDIVIDUAL COMPONENT SCORES
    # --------------------------------------------------------

    df["gearbox_score"] = (
        score_df["GEARBOX"].round(3)
    )

    df["generator_score"] = (
        score_df["GENERATOR"].round(3)
    )

    df["generator_bearing_score"] = (
        score_df["GENERATOR_BEARING"].round(3)
    )

    df["hydraulic_score"] = (
        score_df["HYDRAULIC_GROUP"].round(3)
    )

    df["transformer_score"] = (
        score_df["TRANSFORMER"].round(3)
    )

    # --------------------------------------------------------
    # KEEP ONLY USEFUL OUTPUT COLUMNS
    # --------------------------------------------------------

    output_columns = [
        "Turbine_ID",
        "Timestamp",
        "risk_score",
        "risk_level",

        "predicted_component",
        "component_score",
        "diagnosis_strength",

        "gearbox_score",
        "generator_score",
        "generator_bearing_score",
        "hydraulic_score",
        "transformer_score",
    ]

    output_columns = [
        c for c in output_columns
        if c in df.columns
    ]

    diagnosis_df = df[
        output_columns
    ].copy()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    diagnosis_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        "\nDiagnosis complete."
    )

    print(
        f"Output shape: {diagnosis_df.shape}"
    )

    print(
        f"\nSaved:\n{output_file}"
    )

    print(
        "\nTop diagnosis counts:"
    )

    print(
        diagnosis_df[
            "predicted_component"
        ].value_counts()
    )

    print("\nDiagnosis strength:")

    print(
        diagnosis_df[
            "diagnosis_strength"
        ].value_counts()
    )

    print(
        "\n" + "=" * 80
    )

    print("COMPONENT DIAGNOSIS FINISHED")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()