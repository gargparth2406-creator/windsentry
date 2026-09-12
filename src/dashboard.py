from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd

from nicegui import ui


# ============================================================
# AERIS
# Wind Turbine Intelligence Dashboard
# Part 29 / Part 30
# ============================================================

APP_TITLE = "AERIS | Wind Turbine Intelligence"

BASE_DIR = Path(__file__).resolve().parent.parent

RISK_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "aeris_risk_predictions.csv"
)

HISTORY_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "aeris_historical_patterns.csv"
)


# ============================================================
# BUSINESS CONFIGURATION
# ============================================================

DEFAULT_TARIFF = 6.0

# Dataset sampling interval.
INTERVAL_MINUTES = 10
INTERVAL_HOURS = INTERVAL_MINUTES / 60.0

# These are the same operational temperature thresholds
# used by the dashboard/risk interpretation.
TEMP_WARNING = 66.0
TEMP_HIGH = 68.0

# Operational intervention guidance.
# This is NOT remaining useful life.
ACTION_WINDOWS = {
    "CRITICAL": "Immediate inspection",
    "HIGH": "Within 24 hours",
    "MEDIUM": "Within 7 days",
    "LOW": "Routine monitoring",
}


# ============================================================
# REQUIRED BACKEND COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "Turbine_ID",
    "Timestamp",
    "anomaly_score",
    "is_anomaly",
    "Amb_WindSpeed_Avg",
    "Rtr_RPM_Avg",
    "Gen_RPM_Avg",
    "Gear_Oil_Temp_Avg",
    "Gear_Bear_Temp_Avg",
    "Gen_Bear_Temp_Avg",
    "Hyd_Oil_Temp_Avg",
    "Nac_Temp_Avg",
    "Blds_PitchAngle_Avg",
    "Grd_Prod_Pwr_Avg",
    "expected_power",
    "power_deviation",
    "anomaly_streak",
    "persistent_anomaly",
    "strong_persistent_anomaly",
    "risk_score",
    "risk_level",
]


# ============================================================
# PAGE
# ============================================================

ui.page_title(APP_TITLE)

ui.add_head_html(
    """
    <meta name="theme-color" content="#0b1220">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    """
)


# ============================================================
# GLOBAL CSS
# ============================================================

ui.add_css(
    """
    /* =====================================================
       GLOBAL
       ===================================================== */

    body {
        margin: 0;
        background: #f5f7fa;
        color: #172033;
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    .q-page {
        background: #f5f7fa;
    }

    .aeris-shell {
        width: min(1480px, 94vw);
        margin: 0 auto;
        padding: 26px 0 50px 0;
    }

    .no-shadow {
        box-shadow: none !important;
    }


    /* =====================================================
       HEADER
       ===================================================== */

    .aeris-header {
        background: #0b1220;
        color: white;
        border-radius: 18px;
        padding: 25px 28px;
        margin-bottom: 20px;
        box-shadow: 0 14px 40px rgba(15, 23, 42, 0.12);
    }

    .brand-mark {
        color: #58d0c3;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.22em;
        text-transform: uppercase;
    }

    .header-title {
        color: white;
        font-size: 29px;
        font-weight: 760;
        letter-spacing: -0.035em;
        line-height: 1.1;
        margin-top: 5px;
    }

    .header-subtitle {
        color: #aab6c7;
        font-size: 13px;
        margin-top: 7px;
    }

    .header-meta {
        color: #aab6c7;
        font-size: 12px;
        line-height: 1.75;
        text-align: right;
    }

    .header-meta strong {
        color: white;
        font-weight: 650;
    }


    /* =====================================================
       FILTER BAR
       ===================================================== */

    .filter-bar {
        background: white;
        border: 1px solid #e4e8ee;
        border-radius: 14px;
        padding: 13px 16px;
        margin-bottom: 22px;
        box-shadow: 0 3px 15px rgba(15, 23, 42, 0.035);
    }

    .filter-label {
        color: #667085;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }


    /* =====================================================
       SECTIONS
       ===================================================== */

    .section-title {
        color: #101828;
        font-size: 17px;
        font-weight: 760;
        letter-spacing: -0.015em;
        margin: 27px 0 5px 0;
    }

    .section-caption {
        color: #667085;
        font-size: 12px;
        line-height: 1.55;
        margin-bottom: 13px;
    }


    /* =====================================================
       CARDS
       ===================================================== */

    .aeris-card {
        background: white;
        border: 1px solid #e4e8ee;
        border-radius: 14px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.045);
    }

    .metric-card {
        min-height: 118px;
        padding: 17px 18px;
    }

    .metric-label {
        color: #667085;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .metric-value {
        color: #101828;
        font-size: 26px;
        font-weight: 760;
        letter-spacing: -0.03em;
        margin-top: 7px;
    }

    .metric-note {
        color: #98a2b3;
        font-size: 11px;
        margin-top: 4px;
    }


    /* =====================================================
       STATUS
       ===================================================== */

    .status-card {
        padding: 19px;
        min-height: 185px;
    }

    .eyebrow {
        color: #667085;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.075em;
        text-transform: uppercase;
    }

    .status-heading {
        color: #101828;
        font-size: 24px;
        font-weight: 760;
        letter-spacing: -0.025em;
        margin-top: 4px;
    }

    .status-subtitle {
        color: #667085;
        font-size: 12px;
        margin-top: 3px;
    }

    .status-row {
        border-top: 1px solid #edf0f4;
        padding-top: 10px;
        margin-top: 12px;
    }


    /* =====================================================
       RISK BADGES
       ===================================================== */

    .risk-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 10px;
        font-weight: 850;
        letter-spacing: 0.055em;
    }

    .risk-low {
        background: #ecfdf3;
        color: #067647;
    }

    .risk-medium {
        background: #fff7e6;
        color: #9a6700;
    }

    .risk-high {
        background: #fff1f0;
        color: #b42318;
    }

    .risk-critical {
        background: #f4eaff;
        color: #6927a0;
    }


    /* =====================================================
       PRIORITY BORDERS
       ===================================================== */

    .priority-critical {
        border-left: 4px solid #7f56d9;
    }

    .priority-high {
        border-left: 4px solid #d92d20;
    }

    .priority-medium {
        border-left: 4px solid #f79009;
    }

    .priority-low {
        border-left: 4px solid #12b76a;
    }


    /* =====================================================
       EXPLANATION
       ===================================================== */

    .explanation-card {
        padding: 19px 20px;
    }

    .explanation-heading {
        color: #101828;
        font-size: 14px;
        font-weight: 760;
        margin-bottom: 6px;
    }

    .explanation-text {
        color: #344054;
        font-size: 13px;
        line-height: 1.7;
    }

    .explanation-text + .explanation-heading {
        margin-top: 18px;
    }


    /* =====================================================
       ACTION
       ===================================================== */

    .action-card {
        padding: 18px 20px;
        border-radius: 14px;
        border: 1px solid;
    }

    .action-critical {
        background: #fff5f5;
        border-color: #fecdca;
    }

    .action-high {
        background: #fff8f0;
        border-color: #fedf89;
    }

    .action-medium {
        background: #fffbeb;
        border-color: #f5d98a;
    }

    .action-low {
        background: #effcf5;
        border-color: #abefc6;
    }

    .action-title {
        color: #101828;
        font-size: 13px;
        font-weight: 800;
    }

    .action-body {
        color: #475467;
        font-size: 13px;
        line-height: 1.65;
        margin-top: 5px;
    }


    /* =====================================================
       DATA VALUES
       ===================================================== */

    .data-card {
        padding: 16px;
    }

    .data-label {
        color: #667085;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 0.055em;
        text-transform: uppercase;
    }

    .data-value {
        color: #101828;
        font-size: 19px;
        font-weight: 740;
        margin-top: 5px;
    }

    .data-note {
        color: #98a2b3;
        font-size: 10px;
        margin-top: 3px;
    }


    /* =====================================================
       TABLE
       ===================================================== */

    .q-table__container {
        border-radius: 13px !important;
        border: 1px solid #e4e8ee !important;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.035);
        overflow: hidden;
    }

    .q-table th {
        background: #f8fafc !important;
        color: #667085 !important;
        font-size: 10px !important;
        font-weight: 800 !important;
        letter-spacing: 0.055em;
        text-transform: uppercase;
    }

    .q-table td {
        color: #344054;
        font-size: 12px;
    }


    /* =====================================================
       DRAWER
       ===================================================== */

    .aeris-drawer {
        background: #ffffff;
        border-right: 1px solid #e4e8ee;
    }

    .drawer-inner {
        padding: 22px 18px;
    }

    .drawer-brand {
        color: #101828;
        font-size: 20px;
        font-weight: 820;
        letter-spacing: -0.02em;
    }

    .drawer-caption {
        color: #98a2b3;
        font-size: 11px;
        margin-top: 2px;
    }

    .drawer-section {
        color: #667085;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-top: 21px;
        margin-bottom: 7px;
    }


    /* =====================================================
       CHART CONTAINER
       ===================================================== */

    .chart-card {
        padding: 8px;
        min-height: 350px;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        color: #98a2b3;
        font-size: 10px;
        border-top: 1px solid #e4e8ee;
        margin-top: 35px;
        padding-top: 14px;
    }


    /* =====================================================
       RESPONSIVE
       ===================================================== */

    @media (max-width: 900px) {
        .aeris-shell {
            width: 94vw;
        }

        .header-meta {
            text-align: left;
            margin-top: 14px;
        }

        .header-title {
            font-size: 24px;
        }
    }
    """
)


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    try:
        number = float(value)

        if not np.isfinite(number):
            return default

        return number
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(safe_number(value, default)))
    except Exception:
        return default


def money(value: Any) -> str:
    value = safe_number(value)
    return f"₹{value:,.0f}"


def pct(value: Any) -> str:
    return f"{safe_number(value):.1f}%"


# ============================================================
# DATA LOADING
# ============================================================

def load_risk_data() -> pd.DataFrame:

    if not RISK_FILE.exists():
        raise FileNotFoundError(
            "Risk prediction file was not found:\n"
            f"{RISK_FILE}"
        )

    data = pd.read_csv(RISK_FILE)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Risk prediction file is missing required columns:\n"
            + ", ".join(missing)
        )

    data["Timestamp"] = pd.to_datetime(
        data["Timestamp"],
        errors="coerce",
        utc=True,
    )

    numeric_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in {
            "Turbine_ID",
            "Timestamp",
            "risk_level",
        }
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data["risk_level"] = (
        data["risk_level"]
        .fillna("LOW")
        .astype(str)
        .str.upper()
    )

    data["Turbine_ID"] = (
        data["Turbine_ID"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    data = data.dropna(
        subset=[
            "Timestamp",
            "risk_score",
        ]
    ).copy()

    return (
        data
        .sort_values(
            ["Turbine_ID", "Timestamp"]
        )
        .reset_index(drop=True)
    )


def load_history() -> pd.DataFrame:

    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    history = pd.read_csv(HISTORY_FILE)

    for column in [
        "event_start",
        "event_end",
    ]:

        if column in history.columns:

            history[column] = pd.to_datetime(
                history[column],
                errors="coerce",
                utc=True,
            )

    for column in [
        "duration_minutes",
        "readings",
        "maximum_risk_score",
        "average_risk_score",
        "maximum_power_deviation",
        "minimum_power_deviation",
        "maximum_absolute_power_deviation",
        "maximum_gear_bearing_temperature",
    ]:

        if column in history.columns:
            history[column] = pd.to_numeric(
                history[column],
                errors="coerce",
            )

    if "Turbine_ID" in history.columns:
        history["Turbine_ID"] = (
            history["Turbine_ID"]
            .astype(str)
        )

    return history


try:
    DF = load_risk_data()
    HISTORY = load_history()

except Exception as exc:

    with ui.column().classes("aeris-shell"):

        ui.label(
            "AERIS could not load the backend dataset"
        ).classes(
            "text-h5 text-weight-bold"
        )

        ui.label(
            str(exc)
        ).classes(
            "text-negative"
        )

    ui.run(
        title=APP_TITLE,
        host="127.0.0.1",
        port=8080,
        reload=False,
    )

    raise


# ============================================================
# BUSINESS CALCULATIONS
# ============================================================

def calculate_money_loss(
    row: pd.Series,
    tariff: float,
) -> float:

    deviation = safe_number(
        row.get("power_deviation")
    )

    expected = safe_number(
        row.get("expected_power")
    )

    if deviation >= 0:
        return 0.0

    if expected <= 0:
        return 0.0

    shortfall_kw = (
        expected * abs(deviation)
    )

    lost_kwh = (
        shortfall_kw
        * INTERVAL_HOURS
    )

    return lost_kwh * tariff


def add_business_metrics(
    data: pd.DataFrame,
    tariff: float,
) -> pd.DataFrame:

    result = data.copy()

    result["estimated_money_loss"] = result.apply(
        lambda row: calculate_money_loss(
            row,
            tariff,
        ),
        axis=1,
    )

    result["power_discrepancy_pct"] = (
        result["power_deviation"]
        .abs()
        * 100
    )

    result["power_shortfall_pct"] = (
        (-result["power_deviation"])
        .clip(lower=0)
        * 100
    )

    result["temperature_excess"] = (
        result["Gear_Bear_Temp_Avg"]
        - TEMP_WARNING
    ).clip(lower=0)

    # --------------------------------------------------------
    # PRIORITY
    #
    # 40% financial impact
    # 25% risk
    # 15% power discrepancy
    # 10% persistence
    # 10% temperature
    # --------------------------------------------------------

    loss_reference = (
        result["estimated_money_loss"]
        .quantile(0.95)
    )

    if (
        not np.isfinite(loss_reference)
        or loss_reference <= 0
    ):
        loss_reference = 1.0

    financial_component = (
        result["estimated_money_loss"]
        / loss_reference
    ).clip(0, 1) * 40

    risk_component = (
        result["risk_score"]
        .clip(0, 100)
        / 100
    ) * 25

    discrepancy_component = (
        result["power_shortfall_pct"]
        / 25
    ).clip(0, 1) * 15

    persistence_component = (
        result["persistent_anomaly"]
        .fillna(0)
        .clip(0, 1)
        * 10
    )

    temperature_component = (
        (
            result["Gear_Bear_Temp_Avg"]
            - TEMP_WARNING
        )
        / (TEMP_HIGH - TEMP_WARNING)
    ).clip(0, 1) * 10

    result["priority_score"] = (
        financial_component
        + risk_component
        + discrepancy_component
        + persistence_component
        + temperature_component
    ).clip(0, 100)

    def priority(score):

        score = safe_number(score)

        if score >= 75:
            return "CRITICAL"

        if score >= 50:
            return "HIGH"

        if score >= 25:
            return "MEDIUM"

        return "LOW"

    result["priority"] = (
        result["priority_score"]
        .apply(priority)
    )

    return result


# ============================================================
# HISTORICAL EVENT BUILDER
# ============================================================

def build_current_events(
    data: pd.DataFrame,
) -> pd.DataFrame:

    if data.empty:
        return pd.DataFrame()

    events = []

    for turbine, group in data.groupby(
        "Turbine_ID"
    ):

        group = (
            group
            .sort_values("Timestamp")
            .copy()
        )

        anomaly = (
            group["is_anomaly"]
            .fillna(0)
            .astype(int)
        )

        timestamp_gap = (
            group["Timestamp"]
            .diff()
            .gt(
                pd.Timedelta(
                    minutes=INTERVAL_MINUTES
                )
            )
        )

        starts = (
            (anomaly == 1)
            & (
                (anomaly.shift(
                    1,
                    fill_value=0,
                ) == 0)
                | timestamp_gap
            )
        )

        event_id = starts.cumsum()

        anomalous = group[
            anomaly == 1
        ].copy()

        anomalous["_event_id"] = event_id[
            anomaly == 1
        ].values

        for _, event in anomalous.groupby(
            "_event_id"
        ):

            if event.empty:
                continue

            duration = (
                event["Timestamp"].max()
                - event["Timestamp"].min()
            )

            duration_minutes = (
                safe_number(
                    duration.total_seconds()
                    / 60
                )
                + INTERVAL_MINUTES
            )

            events.append(
                {
                    "Turbine_ID": turbine,
                    "event_start": event[
                        "Timestamp"
                    ].min(),
                    "event_end": event[
                        "Timestamp"
                    ].max(),
                    "duration_minutes": duration_minutes,
                    "readings": len(event),
                    "event_loss": event[
                        "estimated_money_loss"
                    ].sum(),
                    "maximum_risk": event[
                        "risk_score"
                    ].max(),
                    "maximum_temperature": event[
                        "Gear_Bear_Temp_Avg"
                    ].max(),
                    "maximum_power_shortfall": event[
                        "power_shortfall_pct"
                    ].max(),
                }
            )

    return pd.DataFrame(events)


# ============================================================
# HISTORICAL SUMMARY
# ============================================================

def historical_for_selection(
    turbine: str,
) -> pd.DataFrame:

    if HISTORY.empty:
        return pd.DataFrame()

    if turbine == "All turbines":
        return HISTORY.copy()

    if "Turbine_ID" not in HISTORY.columns:
        return pd.DataFrame()

    return HISTORY[
        HISTORY["Turbine_ID"].astype(str)
        == str(turbine)
    ].copy()


def historical_statistics(
    turbine: str,
) -> dict:

    history = historical_for_selection(
        turbine
    )

    if history.empty:
        return {
            "events": 0,
            "median_duration": 0,
            "longest_duration": 0,
            "loss": 0,
        }

    duration = pd.to_numeric(
        history.get(
            "duration_minutes",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )

    return {
        "events": len(history),
        "median_duration": (
            duration.median()
            if not duration.empty
            else 0
        ),
        "longest_duration": (
            duration.max()
            if not duration.empty
            else 0
        ),
    }


# ============================================================
# EXPLANATION ENGINE
# ============================================================

def build_explanation(
    row: pd.Series,
    tariff: float,
    history: pd.DataFrame,
) -> list[tuple[str, str]]:

    turbine = str(
        row["Turbine_ID"]
    )

    risk = str(
        row["risk_level"]
    ).upper()

    deviation = safe_number(
        row["power_deviation"]
    )

    expected = safe_number(
        row["expected_power"]
    )

    actual = safe_number(
        row["Grd_Prod_Pwr_Avg"]
    )

    temperature = safe_number(
        row["Gear_Bear_Temp_Avg"]
    )

    loss = safe_number(
        row["estimated_money_loss"]
    )

    priority = str(
        row["priority"]
    ).upper()

    result = []

    # --------------------------------------------------------
    # WHAT IS WRONG?
    # --------------------------------------------------------

    if deviation < 0 and temperature >= TEMP_HIGH:

        result.append(
            (
                "What's wrong",
                (
                    f"{turbine} is producing approximately "
                    f"{abs(deviation) * 100:.1f}% less power "
                    f"than expected while the gearbox bearing "
                    f"temperature is high at "
                    f"{temperature:.0f} °C."
                ),
            )
        )

    elif deviation < 0:

        result.append(
            (
                "What's wrong",
                (
                    f"{turbine} is producing approximately "
                    f"{abs(deviation) * 100:.1f}% less power "
                    f"than expected."
                ),
            )
        )

    elif temperature >= TEMP_HIGH:

        result.append(
            (
                "What's wrong",
                (
                    f"{turbine} is showing elevated gearbox "
                    f"bearing temperature at "
                    f"{temperature:.0f} °C."
                ),
            )
        )

    elif safe_int(row["is_anomaly"]) == 1:

        result.append(
            (
                "What's wrong",
                (
                    f"{turbine} is showing an abnormal "
                    f"operating pattern."
                ),
            )
        )

    else:

        result.append(
            (
                "What's wrong",
                (
                    f"{turbine} is currently operating without "
                    f"a major abnormal signal in the latest reading."
                ),
            )
        )

    # --------------------------------------------------------
    # WHY DOES IT MATTER?
    # --------------------------------------------------------

    reasons = []

    if safe_int(row["is_anomaly"]) == 1:
        reasons.append(
            "an abnormal operating pattern was detected"
        )

    if safe_int(
        row["persistent_anomaly"]
    ) == 1:
        reasons.append(
            "the abnormal pattern has persisted"
        )

    if safe_int(
        row["strong_persistent_anomaly"]
    ) == 1:
        reasons.append(
            "the persistence is strong"
        )

    if deviation < 0:
        reasons.append(
            f"production is {abs(deviation) * 100:.1f}% below expected"
        )

    if temperature >= TEMP_HIGH:
        reasons.append(
            f"gearbox bearing temperature is {temperature:.0f} °C"
        )

    if reasons:

        result.append(
            (
                "Why it matters",
                (
                    "AERIS is flagging this condition because "
                    + ", ".join(reasons)
                    + "."
                ),
            )
        )

    # --------------------------------------------------------
    # FINANCIAL IMPACT
    # --------------------------------------------------------

    if loss > 0:

        result.append(
            (
                "Financial impact",
                (
                    f"At ₹{tariff:.2f}/kWh, the estimated "
                    f"production revenue loss for this "
                    f"{INTERVAL_MINUTES}-minute reading is "
                    f"{money(loss)}."
                ),
            )
        )

    elif deviation >= 0:

        result.append(
            (
                "Financial impact",
                (
                    "The latest reading does not show an "
                    "estimated production loss because actual "
                    "production is not below expected production."
                ),
            )
        )

    # --------------------------------------------------------
    # TECHNICAL CONTEXT
    # --------------------------------------------------------

    result.append(
        (
            "Operating context",
            (
                f"Actual production is {actual:,.1f} kW versus "
                f"{expected:,.1f} kW expected. "
                f"Current AERIS risk score is "
                f"{safe_number(row['risk_score']):.1f}, "
                f"with {risk} modelled risk."
            ),
        )
    )

    # --------------------------------------------------------
    # HISTORICAL CONTEXT
    # --------------------------------------------------------

    if not history.empty:

        event_count = len(history)

        durations = pd.to_numeric(
            history.get(
                "duration_minutes",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )

        median_duration = (
            durations.median()
            if not durations.empty
            else 0
        )

        if median_duration > 0:

            result.append(
                (
                    "Historical context",
                    (
                        f"{turbine} has {event_count:,} "
                        f"historical risk events in the available "
                        f"history. Similar historical events have "
                        f"a median duration of "
                        f"{median_duration / 60:.1f} hours."
                    ),
                )
            )

    return result


def action_for_priority(
    priority: str,
) -> tuple[str, str]:

    priority = priority.upper()

    if priority == "CRITICAL":

        return (
            "Immediate inspection",
            (
                "Do not treat this as a normal operating condition. "
                "Inspect the turbine as soon as practical and "
                "investigate the source of the abnormal signal."
            ),
            "action-critical",
        )

    if priority == "HIGH":

        return (
            "Inspection within 24 hours",
            (
                "The combination of risk, production discrepancy "
                "and/or financial impact justifies technician "
                "attention within the next 24 hours."
            ),
            "action-high",
        )

    if priority == "MEDIUM":

        return (
            "Review within 7 days",
            (
                "Continue monitoring the turbine and arrange an "
                "inspection within the next week if the condition "
                "persists or worsens."
            ),
            "action-medium",
        )

    return (
        "Routine monitoring",
        (
            "No elevated intervention is indicated by the current "
            "signals. Continue normal monitoring and investigate "
            "if the condition changes."
        ),
        "action-low",
    )


# ============================================================
# UI HELPERS
# ============================================================

def metric_card(
    parent,
    label: str,
    value: str,
    note: str,
):

    with parent:

        with ui.card().classes(
            "aeris-card metric-card flex-1"
        ):

            ui.label(
                label
            ).classes("metric-label")

            ui.label(
                value
            ).classes("metric-value")

            ui.label(
                note
            ).classes("metric-note")


def data_value_card(
    parent,
    label: str,
    value: str,
    note: str = "",
):

    with parent:

        with ui.card().classes(
            "aeris-card data-card flex-1"
        ):

            ui.label(
                label
            ).classes("data-label")

            ui.label(
                value
            ).classes("data-value")

            if note:

                ui.label(
                    note
                ).classes("data-note")


def add_risk_badge(
    level: str,
):

    level = str(
        level
    ).upper()

    css = {
        "LOW": "risk-low",
        "MEDIUM": "risk-medium",
        "HIGH": "risk-high",
        "CRITICAL": "risk-critical",
    }.get(
        level,
        "risk-medium",
    )

    ui.label(
        level
    ).classes(
        f"risk-badge {css}"
    )


# ============================================================
# CHART HELPERS
# ============================================================

def time_labels(data: pd.DataFrame):

    return [
        ts.strftime("%d %b\n%H:%M")
        for ts in data["Timestamp"]
    ]


def base_chart():

    return {
        "backgroundColor": "transparent",
        "animation": False,
        "textStyle": {
            "fontFamily": "Inter, sans-serif",
            "color": "#667085",
        },
        "tooltip": {
            "trigger": "axis",
        },
    }


def build_power_chart(
    data: pd.DataFrame,
):

    chart = base_chart()

    labels = time_labels(data)

    chart.update(
        {
            "legend": {
                "data": [
                    "Actual power",
                    "Expected power",
                ],
                "top": 5,
                "textStyle": {
                    "fontSize": 11,
                    "color": "#667085",
                },
            },
            "grid": {
                "left": 52,
                "right": 22,
                "top": 48,
                "bottom": 62,
            },
            "xAxis": {
                "type": "category",
                "data": labels,
                "axisLabel": {
                    "fontSize": 9,
                    "color": "#98A2B3",
                },
                "axisLine": {
                    "lineStyle": {
                        "color": "#E4E7EC",
                    }
                },
            },
            "yAxis": {
                "type": "value",
                "name": "kW",
                "nameTextStyle": {
                    "fontSize": 10,
                    "color": "#98A2B3",
                },
                "splitLine": {
                    "lineStyle": {
                        "color": "#EEF2F6",
                    }
                },
            },
            "series": [
                {
                    "name": "Actual power",
                    "type": "line",
                    "data": [
                        safe_number(x)
                        for x in data[
                            "Grd_Prod_Pwr_Avg"
                        ]
                    ],
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {
                        "width": 2.5,
                        "color": "#2563EB",
                    },
                    "itemStyle": {
                        "color": "#2563EB",
                    },
                },
                {
                    "name": "Expected power",
                    "type": "line",
                    "data": [
                        safe_number(x)
                        for x in data[
                            "expected_power"
                        ]
                    ],
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {
                        "width": 1.8,
                        "type": "dashed",
                        "color": "#98A2B3",
                    },
                },
            ],
        }
    )

    return chart


def build_risk_chart(
    data: pd.DataFrame,
):

    chart = base_chart()

    labels = time_labels(data)

    chart.update(
        {
            "grid": {
                "left": 50,
                "right": 20,
                "top": 25,
                "bottom": 62,
            },
            "xAxis": {
                "type": "category",
                "data": labels,
                "axisLabel": {
                    "fontSize": 9,
                    "color": "#98A2B3",
                },
            },
            "yAxis": {
                "type": "value",
                "min": 0,
                "max": 100,
                "name": "Risk",
                "splitLine": {
                    "lineStyle": {
                        "color": "#EEF2F6",
                    }
                },
            },
            "series": [
                {
                    "name": "Risk score",
                    "type": "line",
                    "data": [
                        safe_number(x)
                        for x in data[
                            "risk_score"
                        ]
                    ],
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {
                        "width": 2.4,
                        "color": "#D92D20",
                    },
                    "areaStyle": {
                        "color": "rgba(217,45,32,0.08)",
                    },
                    "markLine": {
                        "silent": True,
                        "data": [
                            {
                                "yAxis": 50,
                                "lineStyle": {
                                    "type": "dashed",
                                    "color": "#F79009",
                                },
                            },
                            {
                                "yAxis": 75,
                                "lineStyle": {
                                    "type": "dashed",
                                    "color": "#7F56D9",
                                },
                            },
                        ],
                    },
                }
            ],
        }
    )

    return chart


def build_temperature_chart(
    data: pd.DataFrame,
):

    chart = base_chart()

    labels = time_labels(data)

    temperature = [
        safe_number(x)
        for x in data[
            "Gear_Bear_Temp_Avg"
        ]
    ]

    chart.update(
        {
            "legend": {
                "data": [
                    "Gear bearing",
                    "Warning",
                    "High",
                ],
                "top": 5,
                "textStyle": {
                    "fontSize": 11,
                },
            },
            "grid": {
                "left": 50,
                "right": 20,
                "top": 48,
                "bottom": 62,
            },
            "xAxis": {
                "type": "category",
                "data": labels,
                "axisLabel": {
                    "fontSize": 9,
                },
            },
            "yAxis": {
                "type": "value",
                "name": "°C",
                "splitLine": {
                    "lineStyle": {
                        "color": "#EEF2F6",
                    }
                },
            },
            "series": [
                {
                    "name": "Gear bearing",
                    "type": "line",
                    "data": temperature,
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {
                        "width": 2.5,
                        "color": "#F79009",
                    },
                    "areaStyle": {
                        "color": "rgba(247,144,9,0.08)",
                    },
                },
                {
                    "name": "Warning",
                    "type": "line",
                    "data": [
                        TEMP_WARNING
                    ] * len(labels),
                    "showSymbol": False,
                    "lineStyle": {
                        "type": "dashed",
                        "color": "#D92D20",
                    },
                },
                {
                    "name": "High",
                    "type": "line",
                    "data": [
                        TEMP_HIGH
                    ] * len(labels),
                    "showSymbol": False,
                    "lineStyle": {
                        "type": "dashed",
                        "color": "#7F56D9",
                    },
                },
            ],
        }
    )

    return chart


def build_loss_chart(
    data: pd.DataFrame,
):

    chart = base_chart()

    labels = time_labels(data)

    chart.update(
        {
            "grid": {
                "left": 55,
                "right": 20,
                "top": 25,
                "bottom": 62,
            },
            "xAxis": {
                "type": "category",
                "data": labels,
                "axisLabel": {
                    "fontSize": 9,
                },
            },
            "yAxis": {
                "type": "value",
                "name": "₹",
                "splitLine": {
                    "lineStyle": {
                        "color": "#EEF2F6",
                    }
                },
            },
            "series": [
                {
                    "name": "Estimated loss",
                    "type": "bar",
                    "data": [
                        safe_number(x)
                        for x in data[
                            "estimated_money_loss"
                        ]
                    ],
                    "itemStyle": {
                        "color": "#2563EB",
                        "borderRadius": [
                            3,
                            3,
                            0,
                            0,
                        ],
                    },
                }
            ],
        }
    )

    return chart


def build_discrepancy_chart(
    data: pd.DataFrame,
):

    chart = base_chart()

    labels = time_labels(data)

    values = [
        safe_number(x)
        for x in data[
            "power_deviation"
        ]
    ]

    chart.update(
        {
            "grid": {
                "left": 55,
                "right": 20,
                "top": 25,
                "bottom": 62,
            },
            "xAxis": {
                "type": "category",
                "data": labels,
                "axisLabel": {
                    "fontSize": 9,
                },
            },
            "yAxis": {
                "type": "value",
                "name": "%",
                "axisLabel": {
                    "formatter": "{value}%",
                },
                "splitLine": {
                    "lineStyle": {
                        "color": "#EEF2F6",
                    }
                },
            },
            "series": [
                {
                    "name": "Power deviation",
                    "type": "bar",
                    "data": [
                        round(
                            x * 100,
                            2,
                        )
                        for x in values
                    ],
                    "itemStyle": {
                        "color": "#475467",
                    },
                }
            ],
        }
    )

    return chart


def build_turbine_loss_chart(
    data: pd.DataFrame,
):

    grouped = (
        data.groupby("Turbine_ID")[
            "estimated_money_loss"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    return {
        **base_chart(),
        "grid": {
            "left": 55,
            "right": 20,
            "top": 15,
            "bottom": 40,
        },
        "xAxis": {
            "type": "category",
            "data": grouped.index.tolist(),
        },
        "yAxis": {
            "type": "value",
            "name": "₹",
            "splitLine": {
                "lineStyle": {
                    "color": "#EEF2F6",
                }
            },
        },
        "series": [
            {
                "type": "bar",
                "data": [
                    round(
                        safe_number(x),
                        2,
                    )
                    for x in grouped.values
                ],
                "barMaxWidth": 42,
                "itemStyle": {
                    "color": "#0F766E",
                    "borderRadius": [
                        4,
                        4,
                        0,
                        0,
                    ],
                },
            }
        ],
    }


def build_priority_chart(
    data: pd.DataFrame,
):

    grouped = (
        data.groupby(
            ["Turbine_ID", "priority"]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    turbines = sorted(
        data["Turbine_ID"]
        .astype(str)
        .unique()
        .tolist()
    )

    series = []

    colors = {
        "CRITICAL": "#7F56D9",
        "HIGH": "#D92D20",
        "MEDIUM": "#F79009",
        "LOW": "#12B76A",
    }

    for priority in [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ]:

        series.append(
            {
                "name": priority,
                "type": "bar",
                "stack": "priority",
                "data": [
                    int(
                        grouped.loc[
                            turbine,
                            priority
                        ]
                    )
                    if (
                        turbine in grouped.index
                        and priority in grouped.columns
                    )
                    else 0
                    for turbine in turbines
                ],
                "itemStyle": {
                    "color": colors[priority],
                },
            }
        )

    return {
        **base_chart(),
        "legend": {
            "data": [
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW",
            ],
            "top": 0,
        },
        "grid": {
            "left": 45,
            "right": 20,
            "top": 35,
            "bottom": 40,
        },
        "xAxis": {
            "type": "category",
            "data": turbines,
        },
        "yAxis": {
            "type": "value",
            "splitLine": {
                "lineStyle": {
                    "color": "#EEF2F6",
                }
            },
        },
        "series": series,
    }


# ============================================================
# GLOBAL STATE / CONTROLS
# ============================================================

state = {
    "tariff": DEFAULT_TARIFF,
    "turbine": "All turbines",
}


# ============================================================
# HEADER
# ============================================================

with ui.column().classes(
    "aeris-shell"
):

    with ui.row().classes(
        "aeris-header w-full items-center justify-between"
    ):

        with ui.column().classes(
            "gap-0"
        ):

            ui.label(
                "AERIS"
            ).classes(
                "brand-mark"
            )

            ui.label(
                "Wind Turbine Intelligence"
            ).classes(
                "header-title"
            )

            ui.label(
                "Condition, risk and financial impact in one operating view"
            ).classes(
                "header-subtitle"
            )

        with ui.column().classes(
            "header-meta"
        ):

            ui.label(
                f"Validated records · {len(DF):,}"
            )

            ui.label(
                f"Turbines · {DF['Turbine_ID'].nunique()}"
            )

            latest_timestamp = DF[
                "Timestamp"
            ].max()

            if pd.notna(latest_timestamp):

                ui.label(
                    "Latest · "
                    + latest_timestamp.strftime(
                        "%d %b %Y · %H:%M UTC"
                    )
                )


# ============================================================
# DRAWER
# ============================================================

with ui.left_drawer(
    value=True,
    bordered=True,
).classes(
    "aeris-drawer"
).style(
    "width: 285px;"
):

    with ui.column().classes(
        "drawer-inner w-full"
    ):

        ui.label(
            "AERIS"
        ).classes(
            "drawer-brand"
        )

        ui.label(
            "Operations intelligence"
        ).classes(
            "drawer-caption"
        )

        ui.separator().classes(
            "q-my-md"
        )

        ui.label(
            "Controls"
        ).classes(
            "drawer-section"
        )

        turbine_options = [
            "All turbines"
        ] + sorted(
            DF[
                "Turbine_ID"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        turbine_select = ui.select(
            turbine_options,
            value="All turbines",
            label="Turbine",
        ).classes(
            "w-full"
        )

        ui.label(
            "Business assumption"
        ).classes(
            "drawer-section"
        )

        tariff_input = ui.number(
            label="Electricity tariff (₹/kWh)",
            value=DEFAULT_TARIFF,
            min=0,
            step=0.5,
            format="%.2f",
        ).classes(
            "w-full"
        )

        ui.label(
            "Used only for estimated production revenue loss."
        ).classes(
            "text-caption text-grey-6"
        )

        ui.separator().classes(
            "q-my-md"
        )

        ui.label(
            "Priority model"
        ).classes(
            "drawer-section"
        )

        ui.label(
            "40% financial impact\n"
            "25% risk score\n"
            "15% production discrepancy\n"
            "10% persistence\n"
            "10% temperature"
        ).classes(
            "text-caption text-grey-7"
        )

        ui.label(
            "Intervention timing is operational guidance, not remaining useful life."
        ).classes(
            "text-caption text-grey-6 q-mt-md"
        )


# ============================================================
# MAIN CONTENT
# ============================================================

content = ui.column().classes(
    "aeris-shell"
)


# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard():

    content.clear()

    tariff = safe_number(
        tariff_input.value,
        DEFAULT_TARIFF,
    )

    if tariff < 0:
        tariff = DEFAULT_TARIFF

    selected = (
        turbine_select.value
        or "All turbines"
    )

    state["tariff"] = tariff
    state["turbine"] = selected

    working = add_business_metrics(
        DF,
        tariff,
    )

    if selected != "All turbines":

        working = working[
            working["Turbine_ID"]
            == str(selected)
        ].copy()

    working = (
        working
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    if working.empty:

        with content:

            ui.label(
                "No data available"
            ).classes(
                "text-h5 text-weight-bold"
            )

            ui.label(
                "There are no records for the selected turbine."
            ).classes(
                "text-grey-7"
            )

        return

    # --------------------------------------------------------
    # CURRENT
    # --------------------------------------------------------

    latest = (
        working
        .sort_values("Timestamp")
        .iloc[-1]
    )

    total_loss = working[
        "estimated_money_loss"
    ].sum()

    anomaly_count = int(
        working["is_anomaly"]
        .fillna(0)
        .sum()
    )

    high_priority_count = int(
        working["priority"]
        .isin(
            ["HIGH", "CRITICAL"]
        )
        .sum()
    )

    max_risk = safe_number(
        working["risk_score"].max()
    )

    avg_risk = safe_number(
        working["risk_score"].mean()
    )

    current_priority = str(
        latest["priority"]
    ).upper()

    current_history = historical_for_selection(
        str(latest["Turbine_ID"])
    )

    # ========================================================
    # FILTER / CONTEXT
    # ========================================================

    with content:

        with ui.row().classes(
            "filter-bar w-full items-center"
        ):

            with ui.column().classes(
                "flex-1 gap-0"
            ):

                ui.label(
                    "ACTIVE VIEW"
                ).classes(
                    "filter-label"
                )

                ui.label(
                    "All turbines"
                    if selected == "All turbines"
                    else f"Turbine {selected}"
                ).classes(
                    "text-subtitle1 text-weight-bold"
                )

            with ui.column().classes(
                "flex-1 gap-0"
            ):

                ui.label(
                    "DATA RANGE"
                ).classes(
                    "filter-label"
                )

                ui.label(
                    f"{working['Timestamp'].min().strftime('%d %b %Y')}"
                    " — "
                    f"{working['Timestamp'].max().strftime('%d %b %Y')}"
                ).classes(
                    "text-subtitle1"
                )

            with ui.column().classes(
                "flex-1 gap-0"
            ):

                ui.label(
                    "TARIFF"
                ).classes(
                    "filter-label"
                )

                ui.label(
                    f"₹{tariff:.2f}/kWh"
                ).classes(
                    "text-subtitle1"
                )

        # ====================================================
        # EXECUTIVE SUMMARY
        # ====================================================

        ui.label(
            "Fleet overview"
        ).classes(
            "section-title"
        )

        ui.label(
            "A compact view of production impact, abnormal behaviour and current risk."
        ).classes(
            "section-caption"
        )

        with ui.row().classes(
            "w-full items-stretch"
        ).style(
            "gap: 14px;"
        ):

            metric_card(
                ui.row().classes("contents"),
                "Estimated revenue loss",
                money(total_loss),
                "Across selected records",
            )

            metric_card(
                ui.row().classes("contents"),
                "Anomaly readings",
                f"{anomaly_count:,}",
                "Abnormal observations",
            )

            metric_card(
                ui.row().classes("contents"),
                "High / critical",
                f"{high_priority_count:,}",
                "Financially or operationally important",
            )

            metric_card(
                ui.row().classes("contents"),
                "Maximum risk",
                f"{max_risk:.1f}",
                f"Average risk {avg_risk:.1f}",
            )

        # ====================================================
        # CURRENT CONDITION
        # ====================================================

        ui.label(
            "Current condition"
        ).classes(
            "section-title"
        )

        ui.label(
            "The latest available reading is translated into a human-readable operating assessment."
        ).classes(
            "section-caption"
        )

        with ui.row().classes(
            "w-full items-stretch"
        ).style(
            "gap: 14px;"
        ):

            # ------------------------------------------------
            # CURRENT STATUS CARD
            # ------------------------------------------------

            with ui.card().classes(
                f"aeris-card status-card flex-1 "
                f"priority-{current_priority.lower()}"
            ):

                ui.label(
                    "CURRENT TURBINE"
                ).classes(
                    "eyebrow"
                )

                ui.label(
                    str(latest["Turbine_ID"])
                ).classes(
                    "status-heading"
                )

                ui.label(
                    latest["Timestamp"].strftime(
                        "%d %b %Y · %H:%M UTC"
                    )
                ).classes(
                    "status-subtitle"
                )

                with ui.row().classes(
                    "items-center q-mt-md"
                ).style(
                    "gap: 9px;"
                ):

                    add_risk_badge(
                        latest["risk_level"]
                    )

                    ui.label(
                        f"Priority · {current_priority}"
                    ).classes(
                        "text-caption text-weight-bold"
                    )

                with ui.row().classes(
                    "status-row w-full"
                ):

                    with ui.column().classes(
                        "flex-1 gap-0"
                    ):

                        ui.label(
                            "Risk score"
                        ).classes(
                            "data-label"
                        )

                        ui.label(
                            f"{safe_number(latest['risk_score']):.1f}"
                        ).classes(
                            "data-value"
                        )

                    with ui.column().classes(
                        "flex-1 gap-0"
                    ):

                        ui.label(
                            "Estimated loss"
                        ).classes(
                            "data-label"
                        )

                        ui.label(
                            money(
                                latest[
                                    "estimated_money_loss"
                                ]
                            )
                        ).classes(
                            "data-value"
                        )

            # ------------------------------------------------
            # CONDITION CARD
            # ------------------------------------------------

            with ui.card().classes(
                "aeris-card status-card flex-1"
            ):

                ui.label(
                    "KEY CONDITION"
                ).classes(
                    "eyebrow"
                )

                temperature = safe_number(
                    latest[
                        "Gear_Bear_Temp_Avg"
                    ]
                )

                deviation = safe_number(
                    latest[
                        "power_deviation"
                    ]
                )

                if temperature >= TEMP_HIGH:

                    condition = "Temperature high"

                elif temperature >= TEMP_WARNING:

                    condition = "Temperature elevated"

                elif deviation < -0.05:

                    condition = "Production below expected"

                else:

                    condition = "No major active condition"

                ui.label(
                    condition
                ).classes(
                    "status-heading"
                )

                ui.label(
                    "Latest machine signal"
                ).classes(
                    "status-subtitle"
                )

                with ui.row().classes(
                    "status-row w-full"
                ):

                    data_value_card(
                        ui.row().classes("contents"),
                        "Gear bearing",
                        f"{temperature:.0f} °C",
                        (
                            "High"
                            if temperature >= TEMP_HIGH
                            else (
                                "Elevated"
                                if temperature >= TEMP_WARNING
                                else "Normal range"
                            )
                        ),
                    )

                    data_value_card(
                        ui.row().classes("contents"),
                        "Power discrepancy",
                        pct(
                            abs(deviation)
                        ),
                        (
                            "Below expected"
                            if deviation < 0
                            else "At / above expected"
                        ),
                    )

            # ------------------------------------------------
            # ACTION CARD
            # ------------------------------------------------

            with ui.card().classes(
                "aeris-card status-card flex-1"
            ):

                ui.label(
                    "RECOMMENDED TIMING"
                ).classes(
                    "eyebrow"
                )

                action_title, action_body, action_css = (
                    action_for_priority(
                        current_priority
                    )
                )

                ui.label(
                    action_title
                ).classes(
                    "status-heading"
                )

                ui.label(
                    "Operational guidance"
                ).classes(
                    "status-subtitle"
                )

                ui.label(
                    action_body
                ).classes(
                    "text-body2 text-grey-8 q-mt-md"
                )

        # ====================================================
        # WHAT IS HAPPENING
        # ====================================================

        ui.label(
            "What is happening?"
        ).classes(
            "section-title"
        )

        ui.label(
            "AERIS converts the machine signals into an explanation that does not require knowledge of the model."
        ).classes(
            "section-caption"
        )

        with ui.card().classes(
            "aeris-card explanation-card w-full"
        ):

            explanation_blocks = build_explanation(
                latest,
                tariff,
                current_history,
            )

            for heading, text in explanation_blocks:

                ui.label(
                    heading
                ).classes(
                    "explanation-heading"
                )

                ui.label(
                    text
                ).classes(
                    "explanation-text"
                )

        # ====================================================
        # ACTION
        # ====================================================

        ui.label(
            "What should be done?"
        ).classes(
            "section-title"
        )

        with ui.card().classes(
            f"action-card {action_css} w-full"
        ):

            ui.label(
                action_title
            ).classes(
                "action-title"
            )

            ui.label(
                action_body
            ).classes(
                "action-body"
            )

            if current_priority in {
                "HIGH",
                "CRITICAL",
            }:

                ui.label(
                    "Suggested checks: gearbox/bearing temperature, lubrication condition, abnormal noise or vibration, and visible mechanical issues."
                ).classes(
                    "action-body"
                )

        # ====================================================
        # KEY OPERATING VALUES
        # ====================================================

        ui.label(
            "Operating values"
        ).classes(
            "section-title"
        )

        with ui.row().classes(
            "w-full"
        ).style(
            "gap: 12px;"
        ):

            data_value_card(
                ui.row().classes("contents"),
                "Actual power",
                f"{safe_number(latest['Grd_Prod_Pwr_Avg']):,.1f} kW",
            )

            data_value_card(
                ui.row().classes("contents"),
                "Expected power",
                f"{safe_number(latest['expected_power']):,.1f} kW",
            )

            data_value_card(
                ui.row().classes("contents"),
                "Wind speed",
                f"{safe_number(latest['Amb_WindSpeed_Avg']):.1f} m/s",
            )

            data_value_card(
                ui.row().classes("contents"),
                "Gear oil",
                f"{safe_number(latest['Gear_Oil_Temp_Avg']):.1f} °C",
            )

            data_value_card(
                ui.row().classes("contents"),
                "Generator bearing",
                f"{safe_number(latest['Gen_Bear_Temp_Avg']):.1f} °C",
            )

        # ====================================================
        # PERFORMANCE CHARTS
        # ====================================================

        ui.label(
            "Performance and risk"
        ).classes(
            "section-title"
        )

        ui.label(
            "Recent machine behaviour. The charts use the latest 250 readings in the active selection."
        ).classes(
            "section-caption"
        )

        chart_data = working.tail(
            250
        ).copy()

        with ui.row().classes(
            "w-full items-stretch"
        ).style(
            "gap: 14px;"
        ):

            with ui.card().classes(
                "aeris-card chart-card flex-1"
            ):

                ui.echart(
                    build_power_chart(
                        chart_data
                    )
                ).classes(
                    "w-full"
                ).style(
                    "height: 350px;"
                )

            with ui.card().classes(
                "aeris-card chart-card flex-1"
            ):

                ui.echart(
                    build_risk_chart(
                        chart_data
                    )
                ).classes(
                    "w-full"
                ).style(
                    "height: 350px;"
                )

        with ui.row().classes(
            "w-full items-stretch q-mt-md"
        ).style(
            "gap: 14px;"
        ):

            with ui.card().classes(
                "aeris-card chart-card flex-1"
            ):

                ui.echart(
                    build_temperature_chart(
                        chart_data
                    )
                ).classes(
                    "w-full"
                ).style(
                    "height: 330px;"
                )

            with ui.card().classes(
                "aeris-card chart-card flex-1"
            ):

                ui.echart(
                    build_discrepancy_chart(
                        chart_data
                    )
                ).classes(
                    "w-full"
                ).style(
                    "height: 330px;"
                )

        # ====================================================
        # FINANCIAL ANALYSIS
        # ====================================================

        ui.label(
            "Financial impact"
        ).classes(
            "section-title"
        )

        ui.label(
            "Estimated production revenue loss caused by negative power deviation."
        ).classes(
            "section-caption"
        )

        with ui.row().classes(
            "w-full items-stretch"
        ).style(
            "gap: 14px;"
        ):

            with ui.card().classes(
                "aeris-card chart-card flex-1"
            ):

                ui.echart(
                    build_loss_chart(
                        chart_data
                    )
                ).classes(
                    "w-full"
                ).style(
                    "height: 330px;"
                )

            with ui.card().classes(
                "aeris-card chart-card flex-1"
            ):

                ui.echart(
                    build_turbine_loss_chart(
                        working
                    )
                ).classes(
                    "w-full"
                ).style(
                    "height: 330px;"
                )

        # ====================================================
        # PRIORITY BY TURBINE
        # ====================================================

        ui.label(
            "Priority landscape"
        ).classes(
            "section-title"
        )

        ui.label(
            "This view shows where the largest concentration of operational priorities sits."
        ).classes(
            "section-caption"
        )

        with ui.card().classes(
            "aeris-card chart-card w-full"
        ):

            ui.echart(
                build_priority_chart(
                    working
                )
            ).classes(
                "w-full"
            ).style(
                "height: 350px;"
            )

        # ====================================================
        # PRIORITY QUEUE
        # ====================================================

        ui.label(
            "Priority queue"
        ).classes(
            "section-title"
        )

        ui.label(
            "Conditions ranked primarily by financial impact, then risk, discrepancy, persistence and temperature."
        ).classes(
            "section-caption"
        )

        priority_rows = (
            working[
                working["priority"].isin(
                    [
                        "CRITICAL",
                        "HIGH",
                        "MEDIUM",
                    ]
                )
            ]
            .sort_values(
                [
                    "priority_score",
                    "estimated_money_loss",
                ],
                ascending=False,
            )
            .head(30)
        )

        if priority_rows.empty:

            with ui.card().classes(
                "aeris-card q-pa-md w-full"
            ):

                ui.label(
                    "No elevated-priority conditions are present in the selected data."
                ).classes(
                    "text-positive text-weight-bold"
                )

        else:

            rows = []

            for _, row in priority_rows.iterrows():

                rows.append(
                    {
                        "Turbine": str(
                            row["Turbine_ID"]
                        ),
                        "Timestamp": row[
                            "Timestamp"
                        ].strftime(
                            "%d %b %Y %H:%M"
                        ),
                        "Priority": str(
                            row["priority"]
                        ),
                        "Score": f"{safe_number(row['priority_score']):.1f}",
                        "Risk": f"{safe_number(row['risk_score']):.1f}",
                        "Discrepancy": (
                            f"{safe_number(row['power_shortfall_pct']):.1f}%"
                        ),
                        "Loss": money(
                            row[
                                "estimated_money_loss"
                            ]
                        ),
                        "Gear temp": (
                            f"{safe_number(row['Gear_Bear_Temp_Avg']):.0f} °C"
                        ),
                    }
                )

            ui.table(
                columns=[
                    {
                        "name": "Turbine",
                        "label": "Turbine",
                        "field": "Turbine",
                    },
                    {
                        "name": "Timestamp",
                        "label": "Timestamp",
                        "field": "Timestamp",
                    },
                    {
                        "name": "Priority",
                        "label": "Priority",
                        "field": "Priority",
                    },
                    {
                        "name": "Score",
                        "label": "Priority score",
                        "field": "Score",
                    },
                    {
                        "name": "Risk",
                        "label": "Risk",
                        "field": "Risk",
                    },
                    {
                        "name": "Discrepancy",
                        "label": "Power discrepancy",
                        "field": "Discrepancy",
                    },
                    {
                        "name": "Loss",
                        "label": "Estimated loss",
                        "field": "Loss",
                    },
                    {
                        "name": "Gear temp",
                        "label": "Gear bearing",
                        "field": "Gear temp",
                    },
                ],
                rows=rows,
                row_key="Timestamp",
                pagination={
                    "rowsPerPage": 10
                },
            ).classes(
                "w-full"
            )

        # ====================================================
        # HISTORICAL PATTERNS
        # ====================================================

        ui.label(
            "Historical patterns"
        ).classes(
            "section-title"
        )

        ui.label(
            "Historical risk events generated by Part 28. These are used for context, not as a remaining-useful-life prediction."
        ).classes(
            "section-caption"
        )

        history = historical_for_selection(
            selected
        )

        if not history.empty:

            stats = historical_statistics(
                selected
            )

            with ui.row().classes(
                "w-full"
            ).style(
                "gap: 12px;"
            ):

                data_value_card(
                    ui.row().classes("contents"),
                    "Historical events",
                    f"{stats['events']:,}",
                    "Recorded risk periods",
                )

                data_value_card(
                    ui.row().classes("contents"),
                    "Median event",
                    (
                        f"{stats['median_duration'] / 60:.1f} h"
                        if stats["median_duration"]
                        else "—"
                    ),
                    "Typical historical duration",
                )

                data_value_card(
                    ui.row().classes("contents"),
                    "Longest event",
                    (
                        f"{stats['longest_duration'] / 60:.1f} h"
                        if stats["longest_duration"]
                        else "—"
                    ),
                    "Observed historical maximum",
                )

            # ------------------------------------------------
            # TOP HISTORICAL EVENTS
            # ------------------------------------------------

            display_columns = [
                "Turbine_ID",
                "event_start",
                "duration_minutes",
                "highest_risk_level",
                "maximum_risk_score",
                "maximum_absolute_power_deviation",
                "maximum_gear_bearing_temperature",
            ]

            available = [
                column
                for column in display_columns
                if column in history.columns
            ]

            historical_display = (
                history[
                    available
                ]
                .sort_values(
                    "maximum_risk_score",
                    ascending=False,
                )
                .head(15)
                .copy()
            )

            historical_rows = []

            for _, event in historical_display.iterrows():

                start = event.get(
                    "event_start"
                )

                if pd.notna(start):

                    start_text = start.strftime(
                        "%d %b %Y %H:%M"
                    )

                else:

                    start_text = "—"

                duration = safe_number(
                    event.get(
                        "duration_minutes"
                    )
                )

                historical_rows.append(
                    {
                        "Turbine": str(
                            event.get(
                                "Turbine_ID",
                                "—",
                            )
                        ),
                        "Start": start_text,
                        "Duration": (
                            f"{duration / 60:.1f} h"
                            if duration
                            else "—"
                        ),
                        "Risk": (
                            f"{safe_number(event.get('maximum_risk_score')):.1f}"
                        ),
                        "Power discrepancy": (
                            f"{safe_number(event.get('maximum_absolute_power_deviation')) * 100:.1f}%"
                        ),
                        "Gear bearing": (
                            f"{safe_number(event.get('maximum_gear_bearing_temperature')):.0f} °C"
                        ),
                    }
                )

            if historical_rows:

                ui.table(
                    columns=[
                        {
                            "name": "Turbine",
                            "label": "Turbine",
                            "field": "Turbine",
                        },
                        {
                            "name": "Start",
                            "label": "Start",
                            "field": "Start",
                        },
                        {
                            "name": "Duration",
                            "label": "Duration",
                            "field": "Duration",
                        },
                        {
                            "name": "Risk",
                            "label": "Peak risk",
                            "field": "Risk",
                        },
                        {
                            "name": "Power discrepancy",
                            "label": "Power discrepancy",
                            "field": "Power discrepancy",
                        },
                        {
                            "name": "Gear bearing",
                            "label": "Peak gear bearing",
                            "field": "Gear bearing",
                        },
                    ],
                    rows=historical_rows,
                    row_key="Start",
                    pagination={
                        "rowsPerPage": 10
                    },
                ).classes(
                    "w-full"
                )

        else:

            with ui.card().classes(
                "aeris-card q-pa-md w-full"
            ):

                ui.label(
                    "No historical pattern data is available for this selection."
                ).classes(
                    "text-grey-7"
                )

        # ====================================================
        # INTERPRETATION NOTE
        # ====================================================

        ui.label(
            "Interpretation and assumptions"
        ).classes(
            "section-title"
        )

        with ui.card().classes(
            "aeris-card explanation-card w-full"
        ):

            ui.label(
                "Financial estimate"
            ).classes(
                "explanation-heading"
            )

            ui.label(
                (
                    f"Estimated loss uses negative power deviation, "
                    f"expected power, the {INTERVAL_MINUTES}-minute "
                    f"sampling interval and the configured tariff of "
                    f"₹{tariff:.2f}/kWh. It represents estimated "
                    f"production revenue loss, not an accounting or "
                    f"repair-cost figure."
                )
            ).classes(
                "explanation-text"
            )

            ui.label(
                "Intervention timing"
            ).classes(
                "explanation-heading"
            )

            ui.label(
                (
                    "The dashboard recommends an operational inspection "
                    "window based on the current priority. It does not "
                    "claim that a turbine can safely operate for a specific "
                    "number of days because the available dataset does not "
                    "contain a validated remaining-useful-life model."
                )
            ).classes(
                "explanation-text"
            )

            ui.label(
                "Priority logic"
            ).classes(
                "explanation-heading"
            )

            ui.label(
                (
                    "Priority combines estimated financial impact, AERIS "
                    "risk score, production discrepancy, anomaly persistence "
                    "and gearbox bearing temperature. Financial impact has "
                    "the largest weighting."
                )
            ).classes(
                "explanation-text"
            )

        ui.label(
            "AERIS · Wind Turbine Intelligence · Part 29 / Part 30"
        ).classes(
            "footer"
        )


# ============================================================
# EVENT HANDLERS
# ============================================================

def refresh_dashboard(_=None):

    render_dashboard()


turbine_select.on_value_change(
    refresh_dashboard
)

tariff_input.on_value_change(
    refresh_dashboard
)


# ============================================================
# INITIAL RENDER
# ============================================================

render_dashboard()


# ============================================================
# RUN
# ============================================================

ui.run(
    title=APP_TITLE,
    host="127.0.0.1",
    port=8080,
    reload=False,
)