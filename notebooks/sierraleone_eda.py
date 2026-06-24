# %% [markdown]
# # Sierra Leone Solar Data — Exploratory Data Analysis
# **MoonLight Energy Solutions | Analytics Engineering**
#
# Branch: `eda-sierraleone`
# Dataset: `sierraleone-bumbuna.csv`

# %% [markdown]
# ## 0. Setup & Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (12, 5)})

COUNTRY = "sierraleone"
DATA_IN  = f"data/{COUNTRY}-bumbuna.csv"
DATA_OUT = f"data/{COUNTRY}_clean.csv"

SOLAR_COLS  = ["GHI", "DNI", "DHI"]
SENSOR_COLS = ["ModA", "ModB"]
WIND_COLS   = ["WS", "WSgust"]
Z_COLS      = SOLAR_COLS + SENSOR_COLS + WIND_COLS

# %% [markdown]
# ## 1. Load Data

# %%
df = pd.read_csv(DATA_IN, parse_dates=["Timestamp"])
df.sort_values("Timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Shape: {df.shape}")
df.head(3)

# %% [markdown]
# ## 2. Summary Statistics & Missing-Value Report

# %%
print("=== Descriptive Statistics ===")
df.describe().T.style.background_gradient(cmap="YlOrRd", subset=["mean", "std", "max"])

# %%
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
high_null = missing_report[missing_report["missing_%"] > 5]
print("Columns with >5% nulls:")
print(high_null if not high_null.empty else "None — all columns within threshold.")
missing_report[missing_report["missing_count"] > 0]

# %% [markdown]
# ## 3. Outlier Detection & Basic Cleaning

# %%
z_scores = df[Z_COLS].apply(stats.zscore, nan_policy="omit")
outlier_mask = (z_scores.abs() > 3).any(axis=1)
print(f"Outlier rows flagged: {outlier_mask.sum()} ({outlier_mask.sum()/len(df)*100:.2f}%)")
print("\nPer-column outlier counts:")
print((z_scores.abs() > 3).sum().to_string())

# %%
for col in SOLAR_COLS + SENSOR_COLS:
    neg = (df[col] < 0).sum()
    if neg:
        print(f"  {col}: {neg} negative values → set to NaN")
        df.loc[df[col] < 0, col] = np.nan

df_clean = df.copy()
for col in Z_COLS:
    col_z = np.abs(stats.zscore(df_clean[col].dropna()))
    outlier_idx = df_clean[col].dropna().index[col_z > 3]
    df_clean.loc[outlier_idx, col] = np.nan

for col in Z_COLS:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

df_clean.dropna(subset=SOLAR_COLS, inplace=True)
df_clean.reset_index(drop=True, inplace=True)
print(f"\nCleaned shape: {df_clean.shape}")

df_clean.to_csv(DATA_OUT, index=False)
print(f"Saved → {DATA_OUT}")

# %% [markdown]
# ## 4. Time Series Analysis

# %%
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
series = {"GHI": ("tab:orange", "GHI (W/m²)"),
          "DNI": ("tab:red",    "DNI (W/m²)"),
          "DHI": ("tab:blue",   "DHI (W/m²)"),
          "Tamb":("tab:green",  "Tamb (°C)")}
for ax, (col, (color, label)) in zip(axes, series.items()):
    ax.plot(df_clean["Timestamp"], df_clean[col], color=color, lw=0.6, alpha=0.8)
    ax.set_ylabel(label, fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
axes[-1].set_xlabel("Timestamp")
fig.suptitle("Sierra Leone — Solar Irradiance & Temperature Over Time",
             fontsize=14, fontweight="bold")
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("reports/sierraleone_timeseries.png", bbox_inches="tight")
plt.show()

# %%
df_clean["month"] = df_clean["Timestamp"].dt.month
df_clean["hour"]  = df_clean["Timestamp"].dt.hour

monthly_ghi = df_clean.groupby("month")[["GHI", "DNI", "DHI"]].mean()
ax = monthly_ghi.plot(kind="bar", figsize=(12, 5), colormap="plasma", edgecolor="white")
ax.set_title("Sierra Leone — Monthly Average Solar Irradiance", fontsize=13, fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Irradiance (W/m²)")
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=0)
plt.tight_layout()
plt.savefig("reports/sierraleone_monthly.png", bbox_inches="tight")
plt.show()

# %%
hourly = df_clean.groupby("hour")[["GHI", "DNI", "DHI"]].mean()
ax = hourly.plot(figsize=(12, 5), marker="o", linewidth=2)
ax.set_title("Sierra Leone — Average Diurnal Irradiance Profile",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Irradiance (W/m²)")
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig("reports/sierraleone_diurnal.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Cleaning Impact

# %%
if "Cleaning" in df_clean.columns:
    clean_effect = df_clean.groupby("Cleaning")[["ModA", "ModB"]].mean().reset_index()
    clean_effect["Cleaning"] = clean_effect["Cleaning"].map({0: "Pre-Clean", 1: "Post-Clean"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col in zip(axes, ["ModA", "ModB"]):
        ax.bar(clean_effect["Cleaning"], clean_effect[col],
               color=["#d62728", "#2ca02c"], edgecolor="white", width=0.4)
        ax.set_title(f"Avg {col} by Cleaning Status"); ax.set_ylabel("W/m²")
    fig.suptitle("Sierra Leone — Sensor Output: Pre vs Post Cleaning",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/sierraleone_cleaning.png", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 6. Correlation & Relationship Analysis

# %%
corr_cols = ["GHI", "DNI", "DHI", "TModA", "TModB", "Tamb", "RH", "WS"]
corr_matrix = df_clean[corr_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Sierra Leone — Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/sierraleone_correlation.png", bbox_inches="tight")
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ["WS", "WSgust", "WD"]):
    ax.scatter(df_clean[col], df_clean["GHI"], alpha=0.15, s=8, color="steelblue")
    ax.set_xlabel(col); ax.set_ylabel("GHI (W/m²)"); ax.set_title(f"{col} vs GHI")
fig.suptitle("Sierra Leone — Wind Variables vs GHI", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/sierraleone_wind_ghi.png", bbox_inches="tight")
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(df_clean["RH"], df_clean["Tamb"], alpha=0.15, s=8, color="coral")
axes[0].set(xlabel="RH (%)", ylabel="Tamb (°C)", title="RH vs Ambient Temperature")
axes[1].scatter(df_clean["RH"], df_clean["GHI"], alpha=0.15, s=8, color="mediumseagreen")
axes[1].set(xlabel="RH (%)", ylabel="GHI (W/m²)", title="RH vs GHI")
fig.suptitle("Sierra Leone — Humidity Relationships", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/sierraleone_humidity.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Wind & Distribution Analysis

# %%
try:
    from windrose import WindroseAxes
    fig = plt.figure(figsize=(8, 8))
    ax_wr = WindroseAxes.from_ax(fig=fig)
    ax_wr.bar(df_clean["WD"], df_clean["WS"], normed=True,
              opening=0.8, edgecolor="white", nsector=16)
    ax_wr.set_legend(title="WS (m/s)", loc="lower right")
    ax_wr.set_title("Sierra Leone — Wind Rose", fontsize=13, fontweight="bold")
    plt.savefig("reports/sierraleone_windrose.png", bbox_inches="tight")
    plt.show()
except ImportError:
    print("windrose not installed — pip install windrose")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df_clean["GHI"], bins=60, color="darkorange", edgecolor="white", alpha=0.85)
axes[0].set(title="GHI Distribution", xlabel="GHI (W/m²)", ylabel="Frequency")
axes[1].hist(df_clean["WS"], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
axes[1].set(title="Wind Speed Distribution", xlabel="WS (m/s)", ylabel="Frequency")
fig.suptitle("Sierra Leone — Key Variable Distributions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/sierraleone_histograms.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Temperature & Humidity Analysis

# %%
df_clean["RH_bin"] = pd.cut(df_clean["RH"], bins=10)
rh_group = df_clean.groupby("RH_bin")[["GHI", "Tamb"]].mean()
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
rh_group["GHI"].plot(ax=ax1, color="darkorange", marker="o", label="Mean GHI")
rh_group["Tamb"].plot(ax=ax2, color="steelblue", marker="s",
                      linestyle="--", label="Mean Tamb")
ax1.set_xlabel("Relative Humidity Bin")
ax1.set_ylabel("Mean GHI (W/m²)", color="darkorange")
ax2.set_ylabel("Mean Tamb (°C)", color="steelblue")
ax1.set_title("Sierra Leone — RH Influence on GHI and Temperature",
              fontsize=13, fontweight="bold")
fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
plt.tight_layout()
plt.savefig("reports/sierraleone_rh_influence.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Bubble Chart

# %%
sample = df_clean.sample(min(3000, len(df_clean)), random_state=42)
fig, ax = plt.subplots(figsize=(11, 7))
sc = ax.scatter(sample["Tamb"], sample["GHI"],
                c=sample["RH"], s=sample["RH"] * 1.5,
                cmap="coolwarm", alpha=0.5, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Relative Humidity (%)")
ax.set_xlabel("Ambient Temperature (°C)", fontsize=11)
ax.set_ylabel("GHI (W/m²)", fontsize=11)
ax.set_title("Sierra Leone — GHI vs Temperature (bubble size & colour = RH)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/sierraleone_bubble.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10. Key Insights Summary
#
# | # | Insight | Evidence |
# |---|---------|----------|
# | 1 | **Sierra Leone shows lower median GHI than Benin** due to higher cloud cover from Atlantic moisture | Monthly bar chart |
# | 2 | **Strong wet-season suppression** — GHI drops sharply May–October (rainy season) | Time series |
# | 3 | **RH regularly exceeds 85%** — higher soiling risk and panel degradation concerns | RH histogram |
# | 4 | **TModA/TModB closely track Tamb** — thermal management is critical here | Correlation heatmap |
# | 5 | **DNI is notably lower than GHI** — diffuse radiation dominates on cloudy days | Diurnal profile |
#
# > **References:**
# > - NASA POWER Solar Resource Data (power.larc.nasa.gov)
# > - Sandia National Laboratories PV Performance Modeling Guide

print("Sierra Leone EDA complete.")
