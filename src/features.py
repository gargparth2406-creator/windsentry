import pandas as pd
import numpy as np


def add_time_features(df):
    df = df.copy()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df["hour"] = df["Timestamp"].dt.hour
    df["day"] = df["Timestamp"].dt.day
    df["month"] = df["Timestamp"].dt.month
    df["day_of_week"] = df["Timestamp"].dt.dayofweek

    return df


def add_turbine_features(df):
    df = df.copy()

    # -----------------------------
    # Temperature features
    # -----------------------------

    df["Gen_Temp_Avg"] = (
        df["Gen_Phase1_Temp_Avg"] +
        df["Gen_Phase2_Temp_Avg"] +
        df["Gen_Phase3_Temp_Avg"]
    ) / 3

    df["HVTrafo_Temp_Avg"] = (
        df["HVTrafo_Phase1_Temp_Avg"] +
        df["HVTrafo_Phase2_Temp_Avg"] +
        df["HVTrafo_Phase3_Temp_Avg"]
    ) / 3

    df["Inverter_Temp_Avg"] = (
        df["Grd_RtrInvPhase1_Temp_Avg"] +
        df["Grd_RtrInvPhase2_Temp_Avg"] +
        df["Grd_RtrInvPhase3_Temp_Avg"]
    ) / 3

    # -----------------------------
    # Temperature differences
    # -----------------------------

    df["Gearbox_Temp_Diff"] = (
        df["Gear_Bear_Temp_Avg"] -
        df["Gear_Oil_Temp_Avg"]
    )

    df["Generator_Temp_Diff"] = (
        df["Gen_Bear_Temp_Avg"] -
        df["Amb_Temp_Avg"]
    )

    df["Nacelle_Temp_Diff"] = (
        df["Nac_Temp_Avg"] -
        df["Amb_Temp_Avg"]
    )

    # -----------------------------
    # Vibration / speed features
    # -----------------------------

    df["Gen_RPM_Range"] = (
        df["Gen_RPM_Max"] -
        df["Gen_RPM_Min"]
    )

    df["Rotor_RPM_Range"] = (
        df["Rtr_RPM_Max"] -
        df["Rtr_RPM_Min"]
    )

    # -----------------------------
    # Wind features
    # -----------------------------

    df["WindSpeed_Range"] = (
        df["Amb_WindSpeed_Max"] -
        df["Amb_WindSpeed_Min"]
    )

    # -----------------------------
    # Power features
    # -----------------------------

    df["Power_Range"] = (
        df["Grd_Prod_Pwr_Max"] -
        df["Grd_Prod_Pwr_Min"]
    )

    # -----------------------------
    # Rotor-generator relationship
    # -----------------------------

    df["RPM_Ratio"] = (
        df["Gen_RPM_Avg"] /
        df["Rtr_RPM_Avg"].replace(0, np.nan)
    )

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def create_features(df):
    df = add_time_features(df)
    df = add_turbine_features(df)

    return df


if __name__ == "__main__":

    input_file = "data/processed/edp_combined_raw.csv"
    output_file = "data/processed/edp_features.csv"

    print("Loading data...")

    df = pd.read_csv(input_file)

    print("Original shape:", df.shape)

    df = create_features(df)

    print("Feature engineered shape:", df.shape)

    df.to_csv(output_file, index=False)

    print("Saved:", output_file)