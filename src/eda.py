import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


INPUT_FILE = "data/processed/edp_features.csv"

PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

print("\nDataset shape:")
print(df.shape)

print("\nTurbines:")
print(df["Turbine_ID"].value_counts())

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))


# ---------------------------------------
# 1. Wind Speed vs Power
# ---------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Amb_WindSpeed_Avg"],
    df["Grd_Prod_Pwr_Avg"],
    s=2,
    alpha=0.3
)

plt.xlabel("Wind Speed")
plt.ylabel("Generated Power")
plt.title("Wind Speed vs Generated Power")

plt.savefig(
    PLOT_DIR / "wind_speed_vs_power.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------
# 2. Power distribution
# ---------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    df["Grd_Prod_Pwr_Avg"].dropna(),
    bins=100
)

plt.xlabel("Generated Power")
plt.ylabel("Frequency")
plt.title("Power Distribution")

plt.savefig(
    PLOT_DIR / "power_distribution.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------
# 3. Wind speed distribution
# ---------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    df["Amb_WindSpeed_Avg"].dropna(),
    bins=100
)

plt.xlabel("Wind Speed")
plt.ylabel("Frequency")
plt.title("Wind Speed Distribution")

plt.savefig(
    PLOT_DIR / "wind_speed_distribution.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------
# 4. Generator bearing temperature
# ---------------------------------------

plt.figure(figsize=(12, 6))

for turbine in df["Turbine_ID"].unique():

    temp = df[df["Turbine_ID"] == turbine]

    plt.plot(
        temp["Timestamp"],
        temp["Gen_Bear_Temp_Avg"],
        label=turbine,
        linewidth=0.7
    )

plt.xlabel("Time")
plt.ylabel("Generator Bearing Temperature")
plt.title("Generator Bearing Temperature Over Time")
plt.legend()

plt.savefig(
    PLOT_DIR / "generator_bearing_temperature.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("\nEDA completed.")

print("\nPlots saved in:")
print(PLOT_DIR)