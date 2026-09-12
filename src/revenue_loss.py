import pandas as pd
import numpy as np


# ============================================================
# AERIS ENERGY & REVENUE LOSS ESTIMATOR
# ============================================================

INPUT_FILE = "data/processed/component_diagnosis.csv"
POWER_FILE = "data/processed/aeris_predictions.csv"
OUTPUT_FILE = "data/processed/revenue_loss_predictions.csv"

ELECTRICITY_PRICE_PER_KWH = 8.0


def main():

    print("=" * 80)
    print("AERIS ENERGY & REVENUE LOSS ESTIMATOR")
    print("=" * 80)

    # ========================================================
    # LOAD
    # ========================================================

    print("\nLoading component diagnosis data...")

    df = pd.read_csv(INPUT_FILE)

    print("Loading power prediction data...")

    power_df = pd.read_csv(POWER_FILE)

    print(f"Diagnosis data shape: {df.shape}")
    print(f"Power data shape: {power_df.shape}")

    # ========================================================
    # TIMESTAMPS
    # ========================================================

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        utc=True,
        errors="coerce"
    )

    power_df["Timestamp"] = pd.to_datetime(
        power_df["Timestamp"],
        utc=True,
        errors="coerce"
    )

    # ========================================================
    # KEEP REQUIRED POWER COLUMNS
    # ========================================================

    power_df = power_df[
        [
            "Turbine_ID",
            "Timestamp",
            "Grd_Prod_Pwr_Avg",
            "expected_power",
            "power_deviation"
        ]
    ].copy()

    # ========================================================
    # REMOVE DUPLICATE KEYS
    # ========================================================

    print("\nRemoving duplicate records...")

    diagnosis_before = len(df)
    power_before = len(power_df)

    df = df.drop_duplicates(
        subset=["Turbine_ID", "Timestamp"],
        keep="first"
    ).copy()

    power_df = power_df.drop_duplicates(
        subset=["Turbine_ID", "Timestamp"],
        keep="first"
    ).copy()

    print(
        f"Diagnosis rows: {diagnosis_before} -> {len(df)}"
    )

    print(
        f"Power rows: {power_before} -> {len(power_df)}"
    )

    # ========================================================
    # EXACT MERGE
    # ========================================================

    print("\nMatching power data...")

    df = df.merge(
        power_df,
        on=["Turbine_ID", "Timestamp"],
        how="left"
    )

    # ========================================================
    # CHECK MATCHING
    # ========================================================

    missing_expected = df["expected_power"].isna().sum()

    missing_actual = df["Grd_Prod_Pwr_Avg"].isna().sum()

    print("\nPower-data matching complete.")

    print(
        f"Missing expected_power values: "
        f"{missing_expected}"
    )

    print(
        f"Missing actual power values: "
        f"{missing_actual}"
    )

    print(
        f"Final input shape: {df.shape}"
    )

    # ========================================================
    # POWER LOSS
    # ========================================================

    print("\nCalculating power loss...")

    df["power_loss_kw"] = (
        df["expected_power"]
        - df["Grd_Prod_Pwr_Avg"]
    )

    df["power_loss_kw"] = (
        df["power_loss_kw"]
        .clip(lower=0)
    )

    # ========================================================
    # SAMPLING INTERVAL
    # ========================================================

    print("Determining sampling interval...")

    df = df.sort_values(
        ["Turbine_ID", "Timestamp"]
    ).reset_index(drop=True)

    time_difference = (
        df.groupby("Turbine_ID")["Timestamp"]
        .diff()
        .dt.total_seconds()
        / 3600
    )

    median_interval = (
        time_difference
        .groupby(df["Turbine_ID"])
        .transform("median")
    )

    median_interval = median_interval.fillna(
        10 / 60
    )

    interval_hours = median_interval.clip(
        lower=0,
        upper=1
    )

    df["interval_hours"] = interval_hours

    # ========================================================
    # ENERGY LOSS
    # ========================================================

    df["energy_loss_kwh"] = (
        df["power_loss_kw"]
        * df["interval_hours"]
    )

    # ========================================================
    # REVENUE LOSS
    # ========================================================

    df["revenue_loss"] = (
        df["energy_loss_kwh"]
        * ELECTRICITY_PRICE_PER_KWH
    )

    # ========================================================
    # RISK-WEIGHTED EXPOSURE
    # ========================================================

    if "risk_score" in df.columns:

        risk_factor = (
            pd.to_numeric(
                df["risk_score"],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 100)
            / 100
        )

    else:

        risk_factor = 0

    df["risk_weighted_revenue_exposure"] = (
        df["revenue_loss"]
        * (0.5 + risk_factor)
    )

    # ========================================================
    # 24 HOUR LOSS
    # ========================================================

    df["estimated_24h_energy_loss_kwh"] = (
        df["power_loss_kw"] * 24
    )

    df["estimated_24h_revenue_loss"] = (
        df["estimated_24h_energy_loss_kwh"]
        * ELECTRICITY_PRICE_PER_KWH
    )

    # ========================================================
    # 7 DAY LOSS
    # ========================================================

    df["estimated_7d_energy_loss_kwh"] = (
        df["power_loss_kw"]
        * 24
        * 7
    )

    df["estimated_7d_revenue_loss"] = (
        df["estimated_7d_energy_loss_kwh"]
        * ELECTRICITY_PRICE_PER_KWH
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_columns = [
        "Turbine_ID",
        "Timestamp",

        "risk_score",
        "risk_level",

        "predicted_component",
        "component_score",
        "diagnosis_strength",

        "Grd_Prod_Pwr_Avg",
        "expected_power",
        "power_deviation",

        "power_loss_kw",
        "energy_loss_kwh",
        "revenue_loss",

        "estimated_24h_energy_loss_kwh",
        "estimated_24h_revenue_loss",

        "estimated_7d_energy_loss_kwh",
        "estimated_7d_revenue_loss",

        "risk_weighted_revenue_exposure"
    ]

    output_columns = [
        c
        for c in output_columns
        if c in df.columns
    ]

    result = df[output_columns].copy()

    # ========================================================
    # ROUND
    # ========================================================

    numeric_columns = result.select_dtypes(
        include=np.number
    ).columns

    result[numeric_columns] = (
        result[numeric_columns]
        .round(3)
    )

    # ========================================================
    # SAVE
    # ========================================================

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\nRevenue loss calculation complete.")

    print(
        f"Output shape: {result.shape}"
    )

    print(
        f"\nElectricity price assumption: "
        f"₹{ELECTRICITY_PRICE_PER_KWH}/kWh"
    )

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )

    print("\nTotal estimated historical energy loss:")

    print(
        f"{result['energy_loss_kwh'].sum():,.2f} kWh"
    )

    print("\nTotal estimated historical revenue exposure:")

    print(
        f"₹{result['revenue_loss'].sum():,.2f}"
    )

    # ========================================================
    # TOP 5
    # ========================================================

    print("\nHighest 24-hour revenue exposure:")

    top = result.sort_values(
        "estimated_24h_revenue_loss",
        ascending=False
    ).head(5)

    print(
        top[
            [
                "Turbine_ID",
                "Timestamp",
                "risk_level",
                "predicted_component",
                "estimated_24h_revenue_loss"
            ]
        ].to_string(index=False)
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "ENERGY & REVENUE LOSS ESTIMATION FINISHED"
    )


if __name__ == "__main__":
    main()