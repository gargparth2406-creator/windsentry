import pandas as pd


POWER_FILE = "data/processed/power_predictions.csv"
ANOMALY_FILE = "data/processed/anomaly_predictions.csv"


KEYS = ["Turbine_ID", "Timestamp"]


print("=" * 80)
print("AERIS PREDICTION ALIGNMENT CHECK")
print("=" * 80)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

power = pd.read_csv(POWER_FILE)
anomaly = pd.read_csv(ANOMALY_FILE)


power["Timestamp"] = pd.to_datetime(
    power["Timestamp"],
    utc=True,
    errors="coerce"
)

anomaly["Timestamp"] = pd.to_datetime(
    anomaly["Timestamp"],
    utc=True,
    errors="coerce"
)


print("\nPOWER DATA")
print("Shape:", power.shape)
print("Turbines:", power["Turbine_ID"].nunique())
print(
    "Timestamp range:",
    power["Timestamp"].min(),
    "to",
    power["Timestamp"].max()
)


print("\nANOMALY DATA")
print("Shape:", anomaly.shape)
print("Turbines:", anomaly["Turbine_ID"].nunique())
print(
    "Timestamp range:",
    anomaly["Timestamp"].min(),
    "to",
    anomaly["Timestamp"].max()
)


# ------------------------------------------------------------
# DUPLICATES
# ------------------------------------------------------------

power_duplicates = power.duplicated(
    KEYS
).sum()

anomaly_duplicates = anomaly.duplicated(
    KEYS
).sum()


print("\nDUPLICATE KEYS")

print(
    "Power duplicates:",
    power_duplicates
)

print(
    "Anomaly duplicates:",
    anomaly_duplicates
)


# ------------------------------------------------------------
# KEY SET COMPARISON
# ------------------------------------------------------------

power_keys = set(
    zip(
        power["Turbine_ID"],
        power["Timestamp"]
    )
)

anomaly_keys = set(
    zip(
        anomaly["Turbine_ID"],
        anomaly["Timestamp"]
    )
)


only_in_power = (
    power_keys - anomaly_keys
)

only_in_anomaly = (
    anomaly_keys - power_keys
)

common = (
    power_keys & anomaly_keys
)


print("\nKEY ALIGNMENT")

print(
    "Power keys:",
    len(power_keys)
)

print(
    "Anomaly keys:",
    len(anomaly_keys)
)

print(
    "Common keys:",
    len(common)
)

print(
    "Only in power:",
    len(only_in_power)
)

print(
    "Only in anomaly:",
    len(only_in_anomaly)
)


# ------------------------------------------------------------
# MATCH PERCENTAGE
# ------------------------------------------------------------

if len(anomaly_keys) > 0:

    match_percentage = (
        len(common)
        / len(anomaly_keys)
        * 100
    )

    print(
        f"\nAnomaly → Power match: "
        f"{match_percentage:.2f}%"
    )


if len(power_keys) > 0:

    reverse_percentage = (
        len(common)
        / len(power_keys)
        * 100
    )

    print(
        f"Power → Anomaly match: "
        f"{reverse_percentage:.2f}%"
    )


# ------------------------------------------------------------
# PER-TURBINE COMPARISON
# ------------------------------------------------------------

print("\nPER-TURBINE ALIGNMENT")

turbines = sorted(
    set(
        power["Turbine_ID"].dropna()
    )
    |
    set(
        anomaly["Turbine_ID"].dropna()
    )
)


for turbine in turbines:

    p = power[
        power["Turbine_ID"] == turbine
    ]

    a = anomaly[
        anomaly["Turbine_ID"] == turbine
    ]

    p_keys = set(
        zip(
            p["Turbine_ID"],
            p["Timestamp"]
        )
    )

    a_keys = set(
        zip(
            a["Turbine_ID"],
            a["Timestamp"]
        )
    )

    matched = len(
        p_keys & a_keys
    )

    print(
        f"{turbine}: "
        f"power={len(p_keys)}, "
        f"anomaly={len(a_keys)}, "
        f"matched={matched}"
    )


# ------------------------------------------------------------
# SAMPLE MISMATCHES
# ------------------------------------------------------------

print("\nSAMPLE KEYS ONLY IN ANOMALY")

for key in list(
    only_in_anomaly
)[:10]:

    print(key)


print("\nSAMPLE KEYS ONLY IN POWER")

for key in list(
    only_in_power
)[:10]:

    print(key)


print("\n" + "=" * 80)
print("ALIGNMENT CHECK FINISHED")
print("=" * 80)