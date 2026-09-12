import pandas as pd
import numpy as np


AERIS_FILE = (
    "data/processed/"
    "aeris_risk_predictions.csv"
)

FAILURE_FILE = (
    "data/processed/"
    "failure_log.csv"
)


# ============================================================
# LOAD
# ============================================================

aeris = pd.read_csv(
    AERIS_FILE
)

failures = pd.read_csv(
    FAILURE_FILE
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
# CLEAN
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


# ============================================================
# SIGNAL GROUPS
# ============================================================

COMPONENT_SIGNALS = {

    "GEARBOX": [

        "Gear_Oil_Temp_Avg",

        "Gear_Bear_Temp_Avg",

        "Gen_RPM_Avg",

        "Rtr_RPM_Avg",

        "power_deviation",

        "anomaly_score"

    ],


    "GENERATOR": [

        "Gen_Bear_Temp_Avg",

        "Gen_RPM_Avg",

        "Grd_Prod_Pwr_Avg",

        "power_deviation",

        "anomaly_score"

    ],


    "GENERATOR_BEARING": [

        "Gen_Bear_Temp_Avg",

        "Gen_RPM_Avg",

        "power_deviation",

        "anomaly_score"

    ],


    "TRANSFORMER": [

        "HVTrafo_Phase1_Temp_Avg",

        "HVTrafo_Phase2_Temp_Avg",

        "HVTrafo_Phase3_Temp_Avg",

        "Grd_Prod_Pwr_Avg",

        "power_deviation",

        "anomaly_score"

    ],


    "HYDRAULIC_GROUP": [

        "Hyd_Oil_Temp_Avg",

        "Blds_PitchAngle_Avg",

        "Rtr_RPM_Avg",

        "power_deviation",

        "anomaly_score"

    ]

}


# ============================================================
# ONLY COMMON TURBINES
# ============================================================

failures = failures[
    failures["Turbine_ID"].isin(
        aeris["Turbine_ID"].unique()
    )
].copy()


# ============================================================
# ANALYSIS
# ============================================================

results = []


for _, failure in failures.iterrows():

    turbine = failure["Turbine_ID"]

    failure_time = failure["Timestamp"]

    component = failure["Component"]


    if component not in COMPONENT_SIGNALS:

        continue


    signals = COMPONENT_SIGNALS[
        component
    ]


    turbine_data = aeris[
        aeris["Turbine_ID"] == turbine
    ].copy()


    # --------------------------------------------------------
    # 7 days before failure
    # --------------------------------------------------------

    start = (
        failure_time
        -
        pd.Timedelta(days=7)
    )


    history = turbine_data[
        (
            turbine_data["Timestamp"]
            >= start
        )
        &
        (
            turbine_data["Timestamp"]
            < failure_time
        )
    ].copy()


    if history.empty:

        continue


    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    result = {

        "Turbine_ID":
            turbine,

        "Failure_Time":
            failure_time,

        "Component":
            component,

        "Records":
            len(history)

    }


    for signal in signals:

        if signal not in history.columns:

            continue


        values = history[signal].dropna()


        if values.empty:

            continue


        result[
            f"{signal}_mean"
        ] = values.mean()


        result[
            f"{signal}_max"
        ] = values.max()


        result[
            f"{signal}_min"
        ] = values.min()


        result[
            f"{signal}_std"
        ] = values.std()


        # Last 24 hours

        last_24h = history[
            history["Timestamp"]
            >=
            (
                failure_time
                -
                pd.Timedelta(hours=24)
            )
        ][signal].dropna()


        if not last_24h.empty:

            result[
                f"{signal}_24h_mean"
            ] = last_24h.mean()

            result[
                f"{signal}_24h_max"
            ] = last_24h.max()


    results.append(result)


# ============================================================
# SAVE
# ============================================================

result_df = pd.DataFrame(
    results
)


output = (
    "data/processed/"
    "failure_signatures.csv"
)


result_df.to_csv(
    output,
    index=False
)


# ============================================================
# PRINT
# ============================================================

print("=" * 90)

print(
    "COMPONENT FAILURE SIGNATURE ANALYSIS"
)

print("=" * 90)


print(
    "\nFailures analyzed:",
    len(result_df)
)


print(
    "\nFailure types:"
)

print(
    result_df[
        "Component"
    ].value_counts()
)


print(
    "\nSaved:"
)

print(output)


print(
    "\nFirst results:"
)

print(
    result_df.head(
        20
    ).to_string(
        index=False
    )
)