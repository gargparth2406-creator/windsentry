from nicegui import ui
from pathlib import Path
import pandas as pd


# ============================================================
# AERIS — FINAL DASHBOARD
# Part 29
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RISK_FILE = BASE_DIR / "data" / "processed" / "aeris_risk_predictions.csv"
HISTORY_FILE = BASE_DIR / "data" / "processed" / "aeris_historical_patterns.csv"


# ============================================================
# LOAD EXISTING DATA
# ============================================================

risk_df = pd.read_csv(RISK_FILE)
history_df = pd.read_csv(HISTORY_FILE)

risk_df["Timestamp"] = pd.to_datetime(
    risk_df["Timestamp"],
    errors="coerce",
)

history_df["event_start"] = pd.to_datetime(
    history_df["event_start"],
    errors="coerce",
)

history_df["event_end"] = pd.to_datetime(
    history_df["event_end"],
    errors="coerce",
)


# ============================================================
# BASIC DATA
# ============================================================

turbines = sorted(
    risk_df["Turbine_ID"]
    .dropna()
    .astype(str)
    .unique()
)

latest_timestamp = risk_df["Timestamp"].max()

latest_by_turbine = (
    risk_df
    .sort_values("Timestamp")
    .groupby("Turbine_ID", as_index=False)
    .tail(1)
    .copy()
)

latest_by_turbine["Turbine_ID"] = (
    latest_by_turbine["Turbine_ID"].astype(str)
)


# ============================================================
# COLORS
# ============================================================

NAVY = "#10233f"
NAVY_2 = "#17345d"
BLUE = "#2f6fed"
GREEN = "#17835c"
AMBER = "#b7791f"
RED = "#c63c4a"

BG = "#f4f6f9"
CARD = "#ffffff"
TEXT = "#172033"
MUTED = "#697586"
BORDER = "#e3e8ef"


# ============================================================
# GLOBAL STYLING
# ============================================================

ui.add_head_html(
    f"""
    <style>

        html, body {{
            margin: 0;
            padding: 0;
            background: {BG};
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
            color: {TEXT};
        }}

        * {{
            box-sizing: border-box;
        }}

        .app-shell {{
            min-height: 100vh;
            background: {BG};
        }}

        .topbar {{
            height: 76px;
            background: {NAVY};
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 34px;
            box-shadow: 0 2px 10px rgba(16,35,63,0.16);
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 13px;
        }}

        .brand-mark {{
            width: 38px;
            height: 38px;
            border-radius: 9px;
            background: {BLUE};
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 19px;
            font-weight: 800;
        }}

        .brand-name {{
            font-size: 21px;
            font-weight: 750;
            letter-spacing: 0.03em;
        }}

        .brand-subtitle {{
            font-size: 12px;
            color: #b8c6d9;
            margin-top: 2px;
        }}

        .topbar-status {{
            text-align: right;
            font-size: 12px;
            color: #b8c6d9;
            line-height: 1.5;
        }}

        .topbar-status strong {{
            color: white;
            font-weight: 600;
        }}

        .page {{
            max-width: 1500px;
            margin: 0 auto;
            padding: 30px 34px 50px;
        }}

        .page-title {{
            font-size: 26px;
            font-weight: 720;
            margin: 0;
            color: {TEXT};
        }}

        .page-subtitle {{
            color: {MUTED};
            font-size: 14px;
            margin-top: 5px;
        }}

        .section {{
            margin-top: 28px;
        }}

        .section-heading {{
            font-size: 15px;
            font-weight: 700;
            color: {TEXT};
            margin-bottom: 12px;
        }}

        .section-description {{
            font-size: 12px;
            color: {MUTED};
            margin-top: -7px;
            margin-bottom: 13px;
        }}

        .card {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 12px;
            box-shadow: 0 2px 7px rgba(16,35,63,0.035);
        }}

        .metric-card {{
            padding: 18px 20px;
            min-height: 112px;
        }}

        .metric-label {{
            font-size: 12px;
            color: {MUTED};
            font-weight: 600;
        }}

        .metric-value {{
            font-size: 28px;
            line-height: 1.1;
            font-weight: 750;
            margin-top: 11px;
            color: {TEXT};
        }}

        .metric-note {{
            font-size: 11px;
            color: {MUTED};
            margin-top: 7px;
        }}

        .status-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 15px 18px;
            border-bottom: 1px solid #edf0f4;
        }}

        .status-row:last-child {{
            border-bottom: none;
        }}

        .turbine-id {{
            font-weight: 700;
            font-size: 14px;
            color: {TEXT};
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 68px;
            padding: 5px 11px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 750;
            letter-spacing: 0.04em;
        }}

        .pill-low {{
            background: #e8f5ef;
            color: #16734a;
        }}

        .pill-medium {{
            background: #fff4dd;
            color: #956300;
        }}

        .pill-high {{
            background: #fdebec;
            color: #b42318;
        }}

        .pill-critical {{
            background: #f8dfe2;
            color: #8f1d2c;
        }}

        .score {{
            font-size: 15px;
            font-weight: 700;
            color: {TEXT};
        }}

        .small-muted {{
            color: {MUTED};
            font-size: 12px;
        }}

        .event-card {{
            padding: 17px 19px;
        }}

        .event-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .event-turbine {{
            font-weight: 700;
            font-size: 14px;
        }}

        .event-time {{
            font-size: 11px;
            color: {MUTED};
            margin-top: 4px;
        }}

        .event-score {{
            font-size: 19px;
            font-weight: 750;
            color: {RED};
        }}

        .event-explanation {{
            margin-top: 13px;
            padding-top: 12px;
            border-top: 1px solid #edf0f4;
            color: #465467;
            font-size: 12px;
            line-height: 1.55;
        }}

        .history-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}

        .history-table th {{
            background: #f8fafc;
            color: {MUTED};
            text-align: left;
            font-weight: 650;
            padding: 11px 13px;
            border-bottom: 1px solid {BORDER};
        }}

        .history-table td {{
            padding: 12px 13px;
            border-bottom: 1px solid #edf0f4;
            color: #344054;
        }}

        .history-table tr:last-child td {{
            border-bottom: none;
        }}

        .footer {{
            margin-top: 35px;
            padding-top: 17px;
            border-top: 1px solid {BORDER};
            color: #8a95a5;
            font-size: 11px;
            text-align: center;
        }}

        .q-field__control {{
            border-radius: 8px !important;
        }}

        .q-btn {{
            border-radius: 8px !important;
            text-transform: none !important;
            font-weight: 600;
        }}

    </style>
    """
)


# ============================================================
# HELPERS
# ============================================================

def risk_class(level):
    level = str(level).upper()

    return {
        "LOW": "pill-low",
        "MEDIUM": "pill-medium",
        "HIGH": "pill-high",
        "CRITICAL": "pill-critical",
    }.get(level, "pill-medium")


def latest_status(turbine):
    rows = latest_by_turbine[
        latest_by_turbine["Turbine_ID"] == turbine
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def count_risk(level):
    return int(
        (latest_by_turbine["risk_level"].astype(str).str.upper() == level)
        .sum()
    )


# ============================================================
# HEADER
# ============================================================

with ui.element("div").classes("app-shell"):

    with ui.element("div").classes("topbar"):

        with ui.element("div").classes("brand"):

            ui.label("A").classes("brand-mark")

            with ui.element("div"):
                ui.label("AERIS").classes("brand-name")
                ui.label(
                    "Predictive maintenance intelligence"
                ).classes("brand-subtitle")

        with ui.element("div").classes("topbar-status"):
            ui.html(
                "DATASET STATUS<br>"
                "<strong>Validated prediction data</strong>"
            )
            ui.label(
                f"Latest: {latest_timestamp.strftime('%d %b %Y · %H:%M UTC')}"
            )

    # ========================================================
    # PAGE
    # ========================================================

    with ui.element("main").classes("page"):

        ui.label("Fleet overview").classes("page-title")

        ui.label(
            "Current turbine risk status and historical operating patterns"
        ).classes("page-subtitle")


        # ====================================================
        # FLEET METRICS
        # ====================================================

        with ui.element("section").classes("section"):

            ui.label("Fleet status").classes("section-heading")

            with ui.grid(columns=4).classes("w-full gap-4"):

                with ui.element("div").classes("card metric-card"):
                    ui.label("Turbines monitored").classes("metric-label")
                    ui.label(str(len(turbines))).classes("metric-value")
                    ui.label("Active turbine IDs").classes("metric-note")

                with ui.element("div").classes("card metric-card"):
                    ui.label("High-risk turbines").classes("metric-label")
                    ui.label(str(count_risk("HIGH"))).classes(
                        "metric-value"
                    )
                    ui.label("Latest available reading").classes(
                        "metric-note"
                    )

                with ui.element("div").classes("card metric-card"):
                    anomaly_count = int(
                        latest_by_turbine["is_anomaly"].sum()
                    )
                    ui.label("Current anomalies").classes("metric-label")
                    ui.label(str(anomaly_count)).classes(
                        "metric-value"
                    )
                    ui.label("Across latest turbine readings").classes(
                        "metric-note"
                    )

                with ui.element("div").classes("card metric-card"):
                    ui.label("Historical risk periods").classes(
                        "metric-label"
                    )
                    ui.label(f"{len(history_df):,}").classes(
                        "metric-value"
                    )
                    ui.label("Detected historical periods").classes(
                        "metric-note"
                    )


        # ====================================================
        # TURBINE STATUS + TOP EVENTS
        # ====================================================

        with ui.element("section").classes("section"):

            with ui.grid(columns=2).classes("w-full gap-5"):

                # --------------------------------------------
                # TURBINE STATUS
                # --------------------------------------------

                with ui.element("div").classes("card"):

                    ui.label("Current turbine status").classes(
                        "section-heading"
                    ).style(
                        "padding:18px 18px 0; margin-bottom:0;"
                    )

                    for _, row in latest_by_turbine.sort_values(
                        "risk_score",
                        ascending=False
                    ).iterrows():

                        with ui.element("div").classes("status-row"):

                            with ui.element("div"):
                                ui.label(
                                    str(row["Turbine_ID"])
                                ).classes("turbine-id")

                                ui.label(
                                    pd.to_datetime(
                                        row["Timestamp"]
                                    ).strftime(
                                        "%d %b %Y · %H:%M UTC"
                                    )
                                ).classes("small-muted")

                            with ui.element("div").style(
                                "display:flex; align-items:center; gap:20px;"
                            ):

                                ui.label(
                                    f'{float(row["risk_score"]):.2f}'
                                ).classes("score")

                                ui.label(
                                    str(row["risk_level"]).upper()
                                ).classes(
                                    f'status-pill {risk_class(row["risk_level"])}'
                                )


                # --------------------------------------------
                # TOP HISTORICAL EVENTS
                # --------------------------------------------

                with ui.element("div").classes("card"):

                    ui.label("Highest-risk historical periods").classes(
                        "section-heading"
                    ).style(
                        "padding:18px 18px 0; margin-bottom:0;"
                    )

                    top_events = history_df.sort_values(
                        "maximum_risk_score",
                        ascending=False
                    ).head(5)

                    for _, event in top_events.iterrows():

                        with ui.element("div").classes("event-card"):

                            with ui.element("div").classes("event-top"):

                                with ui.element("div"):

                                    ui.label(
                                        str(event["Turbine_ID"])
                                    ).classes("event-turbine")

                                    ui.label(
                                        pd.to_datetime(
                                            event["event_start"]
                                        ).strftime(
                                            "%d %b %Y · %H:%M UTC"
                                        )
                                    ).classes("event-time")

                                ui.label(
                                    f'{float(event["maximum_risk_score"]):.2f}'
                                ).classes("event-score")

                            ui.label(
                                str(event["explanation"])
                            ).classes("event-explanation")


        # ====================================================
        # TURBINE DETAIL
        # ====================================================

        with ui.element("section").classes("section"):

            ui.label("Turbine detail").classes("section-heading")

            ui.label(
                "Select a turbine to inspect its latest operating condition."
            ).classes("section-description")

            turbine_select = ui.select(
                turbines,
                value=turbines[0] if turbines else None,
                label="Turbine",
            ).classes("w-64")


            detail_container = ui.column().classes("w-full")


            def update_detail():
                detail_container.clear()

                turbine = turbine_select.value

                if not turbine:
                    return

                row = latest_status(turbine)

                if row is None:
                    return

                with detail_container:

                    with ui.grid(columns=4).classes(
                        "w-full gap-4"
                    ):

                        values = [
                            (
                                "Risk level",
                                str(row["risk_level"]).upper(),
                            ),
                            (
                                "Risk score",
                                f'{float(row["risk_score"]):.2f}',
                            ),
                            (
                                "Actual power",
                                f'{float(row["Grd_Prod_Pwr_Avg"]):,.1f} kW',
                            ),
                            (
                                "Expected power",
                                f'{float(row["expected_power"]):,.1f} kW',
                            ),
                            (
                                "Power deviation",
                                f'{float(row["power_deviation"]) * 100:+.2f}%',
                            ),
                            (
                                "Gear bearing temperature",
                                f'{float(row["Gear_Bear_Temp_Avg"]):.0f} °C',
                            ),
                            (
                                "Wind speed",
                                f'{float(row["Amb_WindSpeed_Avg"]):.1f} m/s',
                            ),
                            (
                                "Anomaly",
                                "Detected"
                                if int(row["is_anomaly"]) == 1
                                else "Normal",
                            ),
                        ]

                        for label, value in values:

                            with ui.element("div").classes(
                                "card metric-card"
                            ):

                                ui.label(label).classes(
                                    "metric-label"
                                )

                                ui.label(value).classes(
                                    "metric-value"
                                )


                    ui.label(
                        "Latest reading"
                    ).classes(
                        "section-heading"
                    ).style(
                        "margin-top:24px;"
                    )

                    ui.label(
                        pd.to_datetime(
                            row["Timestamp"]
                        ).strftime(
                            "%d %B %Y · %H:%M UTC"
                        )
                    ).classes("small-muted")


            turbine_select.on_value_change(
                lambda _: update_detail()
            )

            update_detail()


        # ====================================================
        # HISTORICAL PATTERNS
        # ====================================================

        with ui.element("section").classes("section"):

            ui.label("Historical pattern summary").classes(
                "section-heading"
            )

            with ui.element("div").classes("card"):

                summary = (
                    history_df
                    .groupby("Turbine_ID")
                    .agg(
                        periods=("Turbine_ID", "size"),
                        maximum_risk=(
                            "maximum_risk_score",
                            "max",
                        ),
                        maximum_temperature=(
                            "maximum_gear_bearing_temperature",
                            "max",
                        ),
                    )
                    .reset_index()
                    .sort_values(
                        "maximum_risk",
                        ascending=False,
                    )
                )

                table_html = """
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>Turbine</th>
                            <th>Historical periods</th>
                            <th>Maximum risk score</th>
                            <th>Maximum gearbox bearing temperature</th>
                        </tr>
                    </thead>
                    <tbody>
                """

                for _, row in summary.iterrows():

                    table_html += f"""
                        <tr>
                            <td><strong>{row["Turbine_ID"]}</strong></td>
                            <td>{int(row["periods"]):,}</td>
                            <td>{float(row["maximum_risk"]):.2f}</td>
                            <td>{float(row["maximum_temperature"]):.0f} °C</td>
                        </tr>
                    """

                table_html += """
                    </tbody>
                </table>
                """

                ui.html(table_html)


        # ====================================================
        # FOOTER
        # ====================================================

        ui.html(
            """
            <div class="footer">
                AERIS · Predictive maintenance monitoring ·
                Existing validated prediction data
            </div>
            """
        )


# ============================================================
# START APPLICATION
# ============================================================

ui.run(
    title="AERIS | Predictive Maintenance",
    port=8080,
    reload=False,
)