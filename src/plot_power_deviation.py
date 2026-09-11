import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


INPUT_FILE = "data/processed/power_predictions.csv"

PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(exist_ok=True)


df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)


# Select one turbine for visualization
turbine = df["Turbine_ID"].iloc[0]

data = df[
    df["Turbine_ID"] == turbine
].copy()

# Take a manageable portion
data = data.iloc[:3000]


plt.figure(figsize=(14, 6))

plt.plot(
    data["Timestamp"],
    data["Grd_Prod_Pwr_Avg"],
    label="Actual Power"
)

plt.plot(
    data["Timestamp"],
    data["expected_power"],
    label="Expected Power"
)

plt.xlabel("Time")
plt.ylabel("Power")
plt.title(
    f"Actual vs Expected Power - {turbine}"
)

plt.legend()

plt.savefig(
    PLOT_DIR /
    f"{turbine}_actual_vs_expected.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("Plot saved.")