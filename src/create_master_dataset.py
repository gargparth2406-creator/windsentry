import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# AERIS MASTER DATASET BUILDER
# PART 24
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

COMPONENT_FILE = DATA_DIR / "component_diagnosis.csv"
REVENUE_FILE = DATA_DIR / "revenue_loss_predictions.csv"
MAINTENANCE_FILE = DATA_DIR / "maintenance_recommendations.csv"

OUTPUT_FILE = DATA_DIR / "master_aeris_predictions.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_timestamps(df, column="Timestamp"):
    """
    Convert timestamps to a consistent UTC-aware format.
    """

    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' not found."
        )

    df[column] = pd.to_datetime(
        df[column],
        utc=True,
        errors="coerce"
    )

    return df


def remove_duplicate_keys(df, name):
    """
    Ensure one record per Turbine_ID + Timestamp.
    """

    key_columns = [
        "Turbine_ID",
        "Timestamp"
    ]

    before = len(df)

    duplicate_count = df.duplicated(
        subset=key_columns
    ).sum()

    if duplicate_count > 0:

        print(
            f"{name} duplicate keys found: "
            f"{duplicate_count}"
        )

        # Keep the last occurrence.
        df = (
            df.sort_values(
                key_columns
            )
            .drop_duplicates(
                subset=key_columns,
                keep="last"
            )
            .copy()
        )

    after = len(df)

    if before != after:

        print(
            f"{name} rows: "
            f"{before} -> {after}"
        )

    return df


def print_dataset_summary(df):

    print("\n" + "=" * 80)
    print("MASTER DATASET SUMMARY")
    print("=" * 80)

    print(
        f"\nShape: {df.shape}"
    )

    print(
        f"\nUnique turbines: "
        f"{df['Turbine_ID'].nunique()}"
    )

    print(
        f"Unique timestamps: "
        f"{df['Timestamp'].nunique()}"
    )

    print(
        f"Unique turbine-timestamp keys: "
        f"{df[['Turbine_ID', 'Timestamp']].drop_duplicates().shape[0]}"
    )

    print("\nTurbines:")

    print(
        sorted(
            df["Turbine_ID"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    print("\nTimestamp range:")

    print(
        df["Timestamp"].min(),
        "to",
        df["Timestamp"].max()
    )

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    if "risk_level" in df.columns:

        print("\nRisk levels:")

        print(
            df["risk_level"]
            .value_counts(dropna=False)
        )

    # --------------------------------------------------------
    # Component distribution
    # --------------------------------------------------------

    if "predicted_component" in df.columns:

        print("\nPredicted components:")

        print(
            df["predicted_component"]
            .value_counts(dropna=False)
        )

    # --------------------------------------------------------
    # Maintenance priority
    # --------------------------------------------------------

    if "maintenance_priority" in df.columns:

        print("\nMaintenance priorities:")

        print(
            df["maintenance_priority"]
            .value_counts(dropna=False)
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nImportant missing values:")

    important_columns = [
        "risk_score",
        "risk_level",
        "predicted_component",
        "component_score",
        "Grd_Prod_Pwr_Avg",
        "expected_power",
        "power_deviation",
        "power_loss_kw",
        "energy_loss_kwh",
        "revenue_loss",
        "estimated_24h_revenue_loss",
        "estimated_7d_revenue_loss",
        "maintenance_priority",
        "recommended_action",
    ]

    existing_columns = [
        c
        for c in important_columns
        if c in df.columns
    ]

    if existing_columns:

        missing = (
            df[existing_columns]
            .isna()
            .sum()
        )

        print(missing)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AERIS MASTER DATASET BUILDER")
    print("PART 24")
    print("=" * 80)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    print("\nChecking input files...")

    required_files = [
        COMPONENT_FILE,
        REVENUE_FILE,
        MAINTENANCE_FILE,
    ]

    for file in required_files:

        print(
            f"{file.name}: ",
            "FOUND" if file.exists() else "MISSING"
        )

    missing_files = [
        file
        for file in required_files
        if not file.exists()
    ]

    if missing_files:

        raise FileNotFoundError(
            "\nMissing required input files:\n"
            + "\n".join(
                str(file)
                for file in missing_files
            )
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\nLoading component diagnosis data...")

    component = pd.read_csv(
        COMPONENT_FILE
    )

    print(
        "Component shape:",
        component.shape
    )

    print("\nLoading revenue-loss data...")

    revenue = pd.read_csv(
        REVENUE_FILE
    )

    print(
        "Revenue shape:",
        revenue.shape
    )

    print("\nLoading maintenance recommendations...")

    maintenance = pd.read_csv(
        MAINTENANCE_FILE
    )

    print(
        "Maintenance shape:",
        maintenance.shape
    )

    # --------------------------------------------------------
    # NORMALIZE TIMESTAMPS
    # --------------------------------------------------------

    print("\nNormalizing timestamps...")

    component = normalize_timestamps(
        component
    )

    revenue = normalize_timestamps(
        revenue
    )

    maintenance = normalize_timestamps(
        maintenance
    )

    # --------------------------------------------------------
    # BASIC KEY VALIDATION
    # --------------------------------------------------------

    key_columns = [
        "Turbine_ID",
        "Timestamp"
    ]

    for name, data in [
        ("Component", component),
        ("Revenue", revenue),
        ("Maintenance", maintenance),
    ]:

        missing_keys = [
            c
            for c in key_columns
            if c not in data.columns
        ]

        if missing_keys:

            raise ValueError(
                f"{name} dataset is missing "
                f"key columns: {missing_keys}"
            )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    print("\nRemoving duplicate records...")

    component = remove_duplicate_keys(
        component,
        "Component"
    )

    revenue = remove_duplicate_keys(
        revenue,
        "Revenue"
    )

    maintenance = remove_duplicate_keys(
        maintenance,
        "Maintenance"
    )

    # --------------------------------------------------------
    # SELECT COMPONENT COLUMNS
    # --------------------------------------------------------

    component_columns = [
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

    component_columns = [
        c
        for c in component_columns
        if c in component.columns
    ]

    component_master = component[
        component_columns
    ].copy()

    # --------------------------------------------------------
    # SELECT REVENUE / POWER COLUMNS
    # --------------------------------------------------------

    revenue_columns = [
        "Turbine_ID",
        "Timestamp",

        "Grd_Prod_Pwr_Avg",
        "expected_power",
        "power_deviation",

        "power_loss_kw",
        "interval_hours",
        "energy_loss_kwh",
        "revenue_loss",

        "estimated_24h_energy_loss_kwh",
        "estimated_24h_revenue_loss",

        "estimated_7d_energy_loss_kwh",
        "estimated_7d_revenue_loss",

        "risk_weighted_revenue_exposure",
    ]

    revenue_columns = [
        c
        for c in revenue_columns
        if c in revenue.columns
    ]

    revenue_master = revenue[
        revenue_columns
    ].copy()

    # --------------------------------------------------------
    # SELECT MAINTENANCE COLUMNS
    # --------------------------------------------------------

    maintenance_columns = [
        "Turbine_ID",
        "Timestamp",

        "maintenance_priority",
        "recommended_action",
        "priority_reason",
    ]

    # Also preserve any failure-related fields if they
    # exist in the maintenance output.

    optional_maintenance_columns = [
        "maintenance_urgency",
        "component_to_fix",
        "action",
        "recommendation",
    ]

    maintenance_columns += [
        c
        for c in optional_maintenance_columns
        if c in maintenance.columns
    ]

    maintenance_columns = list(
        dict.fromkeys(
            maintenance_columns
        )
    )

    maintenance_columns = [
        c
        for c in maintenance_columns
        if c in maintenance.columns
    ]

    maintenance_master = maintenance[
        maintenance_columns
    ].copy()

    # --------------------------------------------------------
    # MERGE COMPONENT + REVENUE
    # --------------------------------------------------------

    print("\nMerging component diagnosis + revenue data...")

    master = pd.merge(
        component_master,
        revenue_master,
        on=key_columns,
        how="left",
        suffixes=("", "_revenue")
    )

    print(
        "After component/revenue merge:",
        master.shape
    )

    # --------------------------------------------------------
    # MERGE MAINTENANCE
    # --------------------------------------------------------

    print("\nMerging maintenance recommendations...")

    master = pd.merge(
        master,
        maintenance_master,
        on=key_columns,
        how="left",
        suffixes=("", "_maintenance")
    )

    print(
        "After maintenance merge:",
        master.shape
    )

    # --------------------------------------------------------
    # FINAL DUPLICATE CHECK
    # --------------------------------------------------------

    duplicate_keys = master.duplicated(
        subset=key_columns
    ).sum()

    print(
        "\nFinal duplicate key count:",
        duplicate_keys
    )

    if duplicate_keys > 0:

        print(
            "Removing duplicate master records..."
        )

        master = (
            master.sort_values(
                key_columns
            )
            .drop_duplicates(
                subset=key_columns,
                keep="last"
            )
            .copy()
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    master = master.sort_values(
        key_columns
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    numeric_columns = [
        "risk_score",
        "component_score",

        "gearbox_score",
        "generator_score",
        "generator_bearing_score",
        "hydraulic_score",
        "transformer_score",

        "Grd_Prod_Pwr_Avg",
        "expected_power",
        "power_deviation",

        "power_loss_kw",
        "interval_hours",
        "energy_loss_kwh",
        "revenue_loss",

        "estimated_24h_energy_loss_kwh",
        "estimated_24h_revenue_loss",

        "estimated_7d_energy_loss_kwh",
        "estimated_7d_revenue_loss",

        "risk_weighted_revenue_exposure",
    ]

    for column in numeric_columns:

        if column in master.columns:

            master[column] = pd.to_numeric(
                master[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # ROUND NUMERIC DATA
    # --------------------------------------------------------

    existing_numeric = [
        c
        for c in numeric_columns
        if c in master.columns
    ]

    master[existing_numeric] = (
        master[existing_numeric]
        .round(3)
    )

    # --------------------------------------------------------
    # FINAL COLUMN ORDER
    # --------------------------------------------------------

    preferred_order = [
        "Turbine_ID",
        "Timestamp",

        # Diagnosis
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

        # Power
        "Grd_Prod_Pwr_Avg",
        "expected_power",
        "power_deviation",

        # Impact
        "power_loss_kw",
        "interval_hours",
        "energy_loss_kwh",
        "revenue_loss",

        "estimated_24h_energy_loss_kwh",
        "estimated_24h_revenue_loss",

        "estimated_7d_energy_loss_kwh",
        "estimated_7d_revenue_loss",

        "risk_weighted_revenue_exposure",

        # Maintenance
        "maintenance_priority",
        "recommended_action",
        "priority_reason",

        "maintenance_urgency",
        "component_to_fix",
        "action",
        "recommendation",
    ]

    existing_preferred = [
        c
        for c in preferred_order
        if c in master.columns
    ]

    remaining_columns = [
        c
        for c in master.columns
        if c not in existing_preferred
    ]

    master = master[
        existing_preferred
        + remaining_columns
    ]

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    master.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_dataset_summary(
        master
    )

    print("\n" + "=" * 80)
    print("MASTER DATASET CREATED SUCCESSFULLY")
    print("=" * 80)

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nFinal shape:",
        master.shape
    )

    print(
        "\nMaster dataset is ready for the next pipeline stage."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()