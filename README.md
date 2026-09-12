# WindSentry

### Predictive Intelligence for Wind Turbine Fleets

**Team:** Code Crusaders  
**Members:** Parth Garg · Aaditya Sarda · Mustafa Kharodawala · Raag Patel

WindSentry is a **SCADA-driven predictive maintenance platform for wind turbine fleets**. It turns raw turbine telemetry into an explainable diagnosis, a prioritized maintenance action, and an estimate of what it costs to wait.

> **From noisy telemetry to a technician’s next task — explained, prioritized, and priced.**

---

## Problem

Wind turbines can show early signs of degradation before an outright failure. The relevant signals already exist in SCADA telemetry, but operators have to deal with dense, continuously arriving data across many turbines.

WindSentry addresses four core problems:

- Maintenance is often reactive rather than predictive.
- SCADA telemetry is too dense to correlate manually across a fleet.
- Operators need an expected-power baseline to identify underperformance.
- A warning is not enough — teams need to know **why the asset is at risk, how urgent it is, and what the financial impact could be**.

---

## Our Solution

WindSentry builds a continuous diagnosis for every turbine by combining:

1. **Expected-power modelling**
2. **Multi-sensor anomaly detection**
3. **Health scoring**
4. **Failure-risk classification**
5. **Explainable AI**
6. **Root-cause reasoning**
7. **Maintenance prioritization**
8. **Energy-loss estimation**
9. **Revenue-loss estimation**
10. **Technician recommendations**

The central product question is:

> **Is this turbine operating normally — and if not, what should we do about it?**

---

# System Architecture

```text
                    SCADA / SENSOR DATA
                           │
                           ▼
                  ┌──────────────────┐
                  │  Preprocessing   │
                  │ clean / align /  │
                  │ normalize        │
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌──────────────────┐      ┌──────────────────┐
     │ Expected-Power   │      │ Anomaly Model    │
     │ Model            │      │                  │
     │ XGBoost / RF     │      │ Isolation Forest │
     └────────┬─────────┘      └────────┬─────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                  ┌──────────────────┐
                  │   Risk Engine    │
                  │ health + risk    │
                  │ classification   │
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │ SHAP / Explain   │
                  │ feature drivers  │
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │ Root Cause Engine│
                  │ plain-language   │
                  │ fault hypothesis │
                  └────────┬─────────┘
                           ▼
             ┌─────────────────────────────┐
             │ Maintenance Recommendation │
             │ priority · estimated loss │
             │ action · technician        │
             └─────────────────────────────┘
```

The expected-power and anomaly models look at the same telemetry from different perspectives. The risk engine combines the resulting signals before escalating an asset.

---

# Core Capabilities

## 1. SCADA Ingestion

WindSentry consumes timestamped, per-asset telemetry including:

- Wind speed
- Temperature
- Vibration
- Rotor speed
- Power
- Bearing temperature
- Generator temperature

The data is associated with the turbine through an `asset_id`.

## 2. Data Preprocessing

Incoming telemetry is:

- cleaned
- time-aligned
- normalized

before being passed into the predictive models.

## 3. Expected-Power Modelling

A regression model learns the turbine's wind-to-power relationship.

**Model candidates:**

- XGBoost
- Random Forest

The result is the turbine's **expected power at the current operating conditions**.

This creates the baseline required to define underperformance.

## 4. Anomaly Detection

WindSentry uses **Isolation Forest** to detect unusual combinations across the sensor vector.

This allows the system to identify multi-sensor pattern shifts that a single threshold alarm can miss.

## 5. Health Score

Telemetry and model outputs are converted into a continuously trackable **0–100 health score** for each turbine.

## 6. Risk Classification

Assets are assigned a maintenance risk tier using health and failure-probability signals.

Illustrative tiers from the project design:

| Tier | Health Score | Failure Probability | Response |
|---|---:|---:|---|
| Low | 80–100 | < 10% | Routine monitoring |
| Medium | 60–79 | 10–30% | Scheduled inspection |
| High | 35–59 | 30–60% | Prioritized visit |
| Critical | < 35 | > 60% | Immediate dispatch |

> Thresholds are intended to be calibrated per fleet during onboarding.

## 7. Expected vs Actual Power

WindSentry compares the turbine's predicted/expected output with its actual output.

This exposes persistent underperformance and provides the basis for energy-loss estimation.

## 8. Explainable AI

SHAP feature weights show which signals contributed to the current risk score.

Example reasoning:

- Elevated `bearing_temp`
- Increasing vibration variance
- Rotor-speed contribution
- Generator-temperature contribution
- Wind-speed contribution

The goal is to show **evidence behind the score**, not just an unexplained prediction.

## 9. Root-Cause Explanation

Feature contributions are converted into a plain-language fault hypothesis that a technician can act on.

Example:

> Elevated bearing temperature together with rising vibration variance points to bearing wear rather than a generator-side fault.

## 10. Maintenance Priority

Assets are ranked by urgency so maintenance teams know which turbine should be visited first.

The priority view combines the model outputs into an actionable queue.

## 11. Energy-Loss Estimation

The project estimates energy loss from the expected-vs-actual power gap:

```text
Energy Loss (MWh)
= Σ (expected_power − actual_power) × Δt
```

## 12. Revenue-Loss Estimation

Energy loss is translated into financial impact:

```text
Revenue Loss
= Energy Loss × tariff_rate
```

This turns an AI warning into a number that can justify maintenance.

## 13. Interactive Asset View

Each turbine can be explored individually through:

- live/streamed telemetry
- health score
- anomaly score
- failure probability
- expected power
- actual power
- historical context

## 14. Technician Recommendation

A diagnosed fault is converted into a specific recommended action rather than a generic inspection ticket.

Example:

> Bearing inspection within 48 hours, with torque and lubrication checks as the first diagnostic step.

---

# Differentiators

WindSentry is designed to go beyond a conventional predictive-maintenance dashboard.

### Digital-Twin-Style Turbine

A clickable turbine model is bound to sensor channels so the asset is represented as an interactive operational object rather than a static table row.

### Live Sensor Simulation

A synthetic telemetry stream can drive the complete pipeline without waiting for physical hardware.

This makes the system suitable for demonstrations and continuous end-to-end testing.

### Temporal Anomaly Persistence

The system distinguishes between:

- a one-off abnormal reading
- a sustained anomaly
- a worsening condition

by tracking anomaly duration instead of reacting to a single reading.

### Root-Cause Explanation

SHAP contributions are translated into a plain-language explanation that connects detection to an actionable diagnostic hypothesis.

### Human-in-the-Loop Feedback

Technicians can confirm or reject diagnoses. The feedback can be incorporated into future risk scoring and model improvement workflows.

### Maintenance Scheduling

The prioritized risk queue can be converted into an actual maintenance plan for the field team.

### What-If Delay Simulator

Operators can explore the financial and energy consequences of delaying a maintenance action.

Example question:

> **What will it cost if this repair is deferred?**

---

# Data Model

WindSentry is structured around four core tables.

## `assets`

| Field |
|---|
| `asset_id` |
| `type` |
| `location` |
| `capacity` |
| `installation_date` |
| `status` |

## `sensor_readings`

| Field |
|---|
| `timestamp` |
| `asset_id` |
| `wind_speed` |
| `temperature` |
| `vibration` |
| `rotor_speed` |
| `power` |
| `bearing_temp` |
| `generator_temp` |

## `predictions`

| Field |
|---|
| `timestamp` |
| `asset_id` |
| `health_score` |
| `anomaly_score` |
| `failure_probability` |
| `expected_power` |
| `actual_power` |

## `maintenance`

| Field |
|---|
| `asset_id` |
| `priority` |
| `fault` |
| `recommended_action` |
| `estimated_loss` |
| `status` |

Every model output can therefore be traced back to an **asset and timestamp**.

---

# AI / ML Layer

### Expected-Power Model

**XGBoost / Random Forest regression**

**Input:** wind speed, rotor speed, ambient conditions  
**Output:** expected turbine power

### Anomaly Model

**Isolation Forest**

**Input:** full sensor vector  
**Output:** anomaly score

### Explainability

**SHAP**

Used to identify which sensor signals are driving a prediction or risk score.

### Risk Engine

Combines:

- health score
- anomaly signals
- performance gap
- failure probability

to determine the maintenance risk tier.

### Root-Cause Engine

Converts feature contributions into a technician-facing fault hypothesis.

---

# Product Flow

```text
Raw Telemetry
     ↓
Clean & Align
     ↓
Expected Power + Anomaly Detection
     ↓
Health Score
     ↓
Failure Risk
     ↓
Explainable Evidence
     ↓
Root-Cause Hypothesis
     ↓
Maintenance Priority
     ↓
Energy / Revenue Impact
     ↓
Technician Action
```

---

# Build Roadmap

## Phase 1 — Core Pipeline

- SCADA ingestion
- preprocessing
- expected-power model
- anomaly model
- health score
- risk classification

## Phase 2 — Explainability & Impact

- feature importance
- root-cause engine
- energy-loss estimation
- revenue-loss estimation
- maintenance priority

## Phase 3 — Differentiators

- digital-twin-style asset
- live sensor simulation
- temporal anomaly persistence
- human-in-the-loop feedback
- maintenance scheduling
- what-if delay simulator

The core pipeline is designed to stand alone, while the later phases extend it with explainability and interaction.

---

# Why WindSentry?

Traditional monitoring answers:

> **"Is something wrong?"**

WindSentry aims to answer:

> **"Which turbine is at risk, why is it at risk, what will happen if we wait, and what should the technician do next?"**

That is the shift from **alarm monitoring** to **explainable maintenance intelligence**.

---

# Team

### Code Crusaders

- **Mustafa Kharodawala**
- **Parth Garg**
- **Raag Patel**
- **Aaditya Sharda**

---

## Project

**WindSentry — Predictive Intelligence for Wind Turbine Fleets**

**HackOut'26 · Code Crusaders**
