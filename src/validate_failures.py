import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# FILES
# ============================================================

AERIS_FILE = (
    "data/processed/"
    "aeris_risk_predictions.csv"
)

FAILURE_FILE = (
    "data/processed/"
    "failure_log.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "failure_validation.csv"
)


print("=" * 90)
print("AERIS HISTORICAL FAILURE VALIDATION")
print("=" * 90)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading AERIS predictions...")

aeris = pd.read_csv(
    AERIS_FILE
)

print(
    "AERIS shape:",
    aeris.shape
)


print("\nLoading failure log...")

failures = pd.read_csv(
    FAILURE_FILE
)

print(
    "Failure log shape:",
    failures.shape
)


# ============================================================
# DATETIME
# ============================================================

aeris["Timestamp"] = pd.to_datetime(
    aeris["Timestamp"],
    utc=True
).dt.tz_localize(None)


failures["Timestamp"] = pd.to_datetime(
    failures["Timestamp"],
    utc=True
).dt.tz_localize(None)


# ============================================================
# CLEAN IDs
# ============================================================

aeris["Turbine_ID"] = (
    aeris["Turbine_ID"]
    .astype(str)
    .str.strip()
)

failures["Turbine_ID"] = (
    failures["Turbine_ID"]
    .astype(str)
    .str.strip()
)


failures["Component"] = (
    failures["Component"]
    .astype(str)
    .str.strip()
)


# ============================================================
# SORT
# ============================================================

aeris = aeris.sort_values(
    ["Turbine_ID", "Timestamp"]
).reset_index(drop=True)


failures = failures.sort_values(
    ["Turbine_ID", "Timestamp"]
).reset_index(drop=True)


# ============================================================
# FIND COMMON TURBINES
# ============================================================

scada_turbines = set(
    aeris["Turbine_ID"].unique()
)

failure_turbines = set(
    failures["Turbine_ID"].unique()
)

common_turbines = (
    scada_turbines
    &
    failure_turbines
)


print("\nAERIS turbines:")
print(sorted(scada_turbines))


print("\nFailure-log turbines:")
print(sorted(failure_turbines))


print("\nCommon turbines:")
print(sorted(common_turbines))


# ============================================================
# FILTER FAILURES
# ============================================================

usable_failures = failures[
    failures["Turbine_ID"].isin(
        common_turbines
    )
].copy()


print(
    "\nTotal failure events:",
    len(failures)
)

print(
    "Failures with matching AERIS data:",
    len(usable_failures)
)

print(
    "Failures excluded:",
    len(failures) -
    len(usable_failures)
)


# ============================================================
# ANALYZE EACH FAILURE
# ============================================================

results = []


for _, failure in usable_failures.iterrows():

    turbine = failure["Turbine_ID"]

    failure_time = failure["Timestamp"]

    component = failure["Component"]

    remarks = failure["Remarks"]


    # --------------------------------------------------------
    # Turbine history
    # --------------------------------------------------------

    turbine_data = aeris[
        aeris["Turbine_ID"] == turbine
    ].copy()


    # --------------------------------------------------------
    # Basic failure record
    # --------------------------------------------------------

    result = {

        "Turbine_ID": turbine,

        "Failure_Time": failure_time,

        "Component": component,

        "Remarks": remarks

    }


    # ========================================================
    # WINDOWS
    # ========================================================

    windows = {

        "24h": pd.Timedelta(hours=24),

        "3d": pd.Timedelta(days=3),

        "7d": pd.Timedelta(days=7)

    }


    for window_name, duration in windows.items():

        start_time = (
            failure_time - duration
        )


        window = turbine_data[
            (
                turbine_data["Timestamp"]
                >= start_time
            )
            &
            (
                turbine_data["Timestamp"]
                < failure_time
            )
        ].copy()


        # ----------------------------------------------------
        # No data
        # ----------------------------------------------------

        if window.empty:

            result[
                f"{window_name}_records"
            ] = 0

            result[
                f"{window_name}_avg_risk"
            ] = np.nan

            result[
                f"{window_name}_max_risk"
            ] = np.nan

            result[
                f"{window_name}_high_risk_rate"
            ] = np.nan

            result[
                f"{window_name}_critical_risk_rate"
            ] = np.nan

            result[
                f"{window_name}_anomaly_rate"
            ] = np.nan

            result[
                f"{window_name}_persistent_rate"
            ] = np.nan

            result[
                f"{window_name}_strong_persistent_rate"
            ] = np.nan

            result[
                f"{window_name}_avg_power_deviation"
            ] = np.nan

            continue


        # ----------------------------------------------------
        # Record count
        # ----------------------------------------------------

        result[
            f"{window_name}_records"
        ] = len(window)


        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        result[
            f"{window_name}_avg_risk"
        ] = window[
            "risk_score"
        ].mean()


        result[
            f"{window_name}_max_risk"
        ] = window[
            "risk_score"
        ].max()


        # ----------------------------------------------------
        # HIGH risk
        # ----------------------------------------------------

        result[
            f"{window_name}_high_risk_rate"
        ] = (
            window["risk_score"] >= 50
        ).mean()


        # ----------------------------------------------------
        # CRITICAL risk
        # ----------------------------------------------------

        result[
            f"{window_name}_critical_risk_rate"
        ] = (
            window["risk_score"] >= 75
        ).mean()


        # ----------------------------------------------------
        # Anomaly
        # ----------------------------------------------------

        result[
            f"{window_name}_anomaly_rate"
        ] = window[
            "is_anomaly"
        ].mean()


        # ----------------------------------------------------
        # Persistent anomaly
        # ----------------------------------------------------

        result[
            f"{window_name}_persistent_rate"
        ] = window[
            "persistent_anomaly"
        ].mean()


        # ----------------------------------------------------
        # Strong persistent anomaly
        # ----------------------------------------------------

        result[
            f"{window_name}_strong_persistent_rate"
        ] = window[
            "strong_persistent_anomaly"
        ].mean()


        # ----------------------------------------------------
        # Power deviation
        # ----------------------------------------------------

        result[
            f"{window_name}_avg_power_deviation"
        ] = window[
            "power_deviation"
        ].mean()


    results.append(result)


# ============================================================
# CREATE RESULT
# ============================================================

validation = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

Path(
    "data/processed"
).mkdir(
    parents=True,
    exist_ok=True
)


validation.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("FAILURE VALIDATION RESULTS")
print("=" * 90)


if validation.empty:

    print(
        "\nNo failures could be validated."
    )

else:

    display_columns = [

        "Turbine_ID",

        "Failure_Time",

        "Component",

        "24h_avg_risk",

        "3d_avg_risk",

        "7d_avg_risk",

        "7d_max_risk",

        "7d_high_risk_rate",

        "7d_critical_risk_rate",

        "7d_anomaly_rate",

        "7d_persistent_rate",

        "7d_avg_power_deviation"

    ]


    print(
        validation[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("SUMMARY")
print("=" * 90)


if not validation.empty:

    print(
        "\nFailures analyzed:",
        len(validation)
    )


    # --------------------------------------------------------
    # High risk before failure
    # --------------------------------------------------------

    high_risk_failures = validation[
        validation[
            "7d_max_risk"
        ] >= 50
    ]


    print(
        "\nFailures with HIGH risk "
        "at some point in previous 7 days:",
        len(high_risk_failures)
    )


    high_rate = (
        len(high_risk_failures)
        /
        len(validation)
        *
        100
    )


    print(
        f"HIGH-risk detection rate: "
        f"{high_rate:.1f}%"
    )


    # --------------------------------------------------------
    # Critical risk
    # --------------------------------------------------------

    critical_failures = validation[
        validation[
            "7d_max_risk"
        ] >= 75
    ]


    print(
        "\nFailures with CRITICAL risk "
        "at some point in previous 7 days:",
        len(critical_failures)
    )


    critical_rate = (
        len(critical_failures)
        /
        len(validation)
        *
        100
    )


    print(
        f"CRITICAL-risk detection rate: "
        f"{critical_rate:.1f}%"
    )


    # --------------------------------------------------------
    # Persistent anomaly
    # --------------------------------------------------------

    persistent_failures = validation[
        validation[
            "7d_persistent_rate"
        ] > 0
    ]


    print(
        "\nFailures preceded by "
        "persistent anomaly:",
        len(persistent_failures)
    )


    # --------------------------------------------------------
    # Average risk
    # --------------------------------------------------------

    print(
        "\nAverage 7-day risk before failures:",
        round(
            validation[
                "7d_avg_risk"
            ].mean(),
            2
        )
    )


    print(
        "\nAverage maximum risk before failures:",
        round(
            validation[
                "7d_max_risk"
            ].mean(),
            2
        )
    )


print("\n")
print("=" * 90)

print(
    "Saved validation file:"
)

print(
    OUTPUT_FILE
)

print("=" * 90)