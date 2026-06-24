"""
app/utils.py
------------
Data processing and visualization utility functions for the
MoonLight Energy Solutions Streamlit dashboard.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ── Palette ───────────────────────────────────────────────────────────────────
COUNTRY_COLORS = {
    "Benin":        "#E63946",
    "Sierra Leone": "#457B9D",
    "Togo":         "#2A9D8F",
}
IRRADIANCE_COLS = ["GHI", "DNI", "DHI"]
Z_SCORE_COLS    = ["GHI", "DNI", "DHI", "ModA", "ModB", "WS", "WSgust"]


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_country(path: str, country_name: str) -> pd.DataFrame:
    """Load a cleaned CSV and attach a Country label column."""
    df = pd.read_csv(path)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.set_index("Timestamp").sort_index()
    df["Country"] = country_name
    return df


def load_all(data_dir: str = "data") -> dict[str, pd.DataFrame]:
    """Return a dict of {country_label: DataFrame} for all available CSVs."""
    mapping = {
        "Benin":        os.path.join(data_dir, "benin_clean.csv"),
        "Sierra Leone": os.path.join(data_dir, "sierraleone_clean.csv"),
        "Togo":         os.path.join(data_dir, "togo_clean.csv"),
    }
    frames = {}
    for name, path in mapping.items():
        if os.path.exists(path):
            frames[name] = load_country(path, name)
    return frames


# ── Summary Statistics ────────────────────────────────────────────────────────
def summary_table(frames: dict) -> pd.DataFrame:
    """
    Build a cross-country summary table:
    mean, median, std for GHI, DNI, DHI.
    """
    rows = []
    for country, df in frames.items():
        for col in IRRADIANCE_COLS:
            if col in df.columns:
                rows.append({
                    "Country": country,
                    "Metric":  col,
                    "Mean":    round(df[col].mean(), 2),
                    "Median":  round(df[col].median(), 2),
                    "Std Dev": round(df[col].std(), 2),
                    "Max":     round(df[col].max(), 2),
                })
    return pd.DataFrame(rows)


# ── Outlier Detection ─────────────────────────────────────────────────────────
def flag_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Return df with a boolean column 'is_outlier'."""
    cols  = [c for c in Z_SCORE_COLS if c in df.columns]
    z     = np.abs(stats.zscore(df[cols].fillna(df[cols].median())))
    df    = df.copy()
    df["is_outlier"] = (z > threshold).any(axis=1)
    return df


# ── Time Series ───────────────────────────────────────────────────────────────
def plot_time_series(df: pd.DataFrame, country: str,
                     resample: str = "D") -> go.Figure:
    """
    Interactive Plotly time-series of GHI, DNI, DHI, Tamb.
    resample: pandas offset alias — 'H', 'D', 'W', 'M'
    """
    cols = [c for c in ["GHI", "DNI", "DHI", "Tamb"] if c in df.columns]
    agg  = df[cols].resample(resample).mean()

    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True,
                        subplot_titles=cols)
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#F4A261"]
    for i, col in enumerate(cols, 1):
        fig.add_trace(go.Scatter(x=agg.index, y=agg[col],
                                 name=col, line=dict(color=colors[i-1], width=1)),
                      row=i, col=1)
    fig.update_layout(
        title=f"{country} — Solar & Temperature Time Series ({resample} avg)",
        height=180 * len(cols),
        showlegend=True,
        template="plotly_white",
    )
    return fig


def plot_monthly_heatmap(df: pd.DataFrame, col: str = "GHI") -> go.Figure:
    """Hour-of-day × Month pivot heatmap for a given column."""
    d = df[[col]].copy()
    d["Hour"]  = d.index.hour
    d["Month"] = d.index.month
    pivot = d.groupby(["Hour", "Month"])[col].mean().unstack()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="Hour of Day", color=f"{col} (W/m²)"),
        x=[month_labels[m-1] for m in pivot.columns],
        color_continuous_scale="YlOrRd",
        title=f"Average {col} by Hour & Month",
    )
    fig.update_layout(template="plotly_white")
    return fig


# ── Cleaning Impact ───────────────────────────────────────────────────────────
def plot_cleaning_impact(df: pd.DataFrame, country: str) -> go.Figure:
    """Bar chart of average ModA & ModB by Cleaning flag."""
    if "Cleaning" not in df.columns:
        return go.Figure()
    grp = df.groupby("Cleaning")[["ModA", "ModB"]].mean().reset_index()
    grp["Cleaning"] = grp["Cleaning"].map({0: "Not Cleaned", 1: "Cleaned"})
    fig = px.bar(
        grp.melt(id_vars="Cleaning", var_name="Sensor", value_name="Avg (W/m²)"),
        x="Cleaning", y="Avg (W/m²)", color="Sensor", barmode="group",
        title=f"{country} — Average Sensor Reading Pre/Post Cleaning",
        color_discrete_map={"ModA": "#E63946", "ModB": "#457B9D"},
    )
    fig.update_layout(template="plotly_white")
    return fig


# ── Correlation ───────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, country: str) -> plt.Figure:
    """Seaborn heatmap of key irradiance + module temperature correlations."""
    cols = [c for c in ["GHI","DNI","DHI","TModA","TModB","Tamb","RH","WS"]
            if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5, square=True)
    ax.set_title(f"{country} — Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_scatter_wind_ghi(df: pd.DataFrame, country: str) -> go.Figure:
    """Scatter: WS and WSgust vs GHI."""
    cols  = [c for c in ["WS", "WSgust"] if c in df.columns]
    fig   = make_subplots(rows=1, cols=len(cols),
                          subplot_titles=[f"{c} vs GHI" for c in cols])
    colors = ["#457B9D", "#E63946"]
    for i, col in enumerate(cols, 1):
        sample = df[[col, "GHI"]].dropna().sample(min(2000, len(df)), random_state=42)
        fig.add_trace(
            go.Scatter(x=sample[col], y=sample["GHI"], mode="markers",
                       marker=dict(color=colors[i-1], opacity=0.4, size=3),
                       name=col),
            row=1, col=i,
        )
    fig.update_layout(title=f"{country} — Wind Speed vs GHI",
                      height=420, template="plotly_white")
    return fig


def plot_rh_vs_tamb(df: pd.DataFrame, country: str) -> go.Figure:
    """Scatter: RH vs Tamb, colored by GHI."""
    sample = df[["RH","Tamb","GHI"]].dropna().sample(
        min(3000, len(df)), random_state=1)
    fig = px.scatter(sample, x="Tamb", y="RH", color="GHI",
                     color_continuous_scale="YlOrRd",
                     labels={"Tamb": "Ambient Temp (°C)",
                             "RH":   "Relative Humidity (%)",
                             "GHI":  "GHI (W/m²)"},
                     title=f"{country} — RH vs Tamb (colored by GHI)",
                     opacity=0.5)
    fig.update_layout(template="plotly_white")
    return fig


# ── Wind Rose ─────────────────────────────────────────────────────────────────
def plot_wind_rose_plotly(df: pd.DataFrame, country: str) -> go.Figure:
    """Radial bar wind rose using Plotly (no windrose lib needed in Streamlit)."""
    if "WD" not in df.columns or "WS" not in df.columns:
        return go.Figure()
    d = df[["WD", "WS"]].dropna()
    bins   = np.arange(0, 361, 22.5)
    labels = [f"{int(b)}°" for b in bins[:-1]]
    d["sector"] = pd.cut(d["WD"], bins=bins, labels=labels, include_lowest=True)
    agg = d.groupby("sector", observed=True)["WS"].mean().reset_index()
    fig = go.Figure(go.Barpolar(
        r=agg["WS"], theta=agg["sector"],
        marker_color=agg["WS"],
        marker_colorscale="Blues",
        opacity=0.85,
    ))
    fig.update_layout(
        title=f"{country} — Wind Rose (avg WS by direction)",
        polar=dict(radialaxis=dict(visible=True)),
        template="plotly_white",
        height=500,
    )
    return fig


# ── Histograms ────────────────────────────────────────────────────────────────
def plot_histograms(df: pd.DataFrame, country: str,
                    cols: list[str] = None) -> go.Figure:
    """Overlaid histograms for selected columns."""
    if cols is None:
        cols = [c for c in ["GHI", "WS"] if c in df.columns]
    fig = make_subplots(rows=1, cols=len(cols),
                        subplot_titles=[f"{c} Distribution" for c in cols])
    colors = ["#E63946", "#457B9D", "#2A9D8F"]
    for i, col in enumerate(cols, 1):
        fig.add_trace(
            go.Histogram(x=df[col].dropna(), name=col,
                         marker_color=colors[i-1], opacity=0.75,
                         nbinsx=60),
            row=1, col=i,
        )
    fig.update_layout(title=f"{country} — Distributions",
                      template="plotly_white", showlegend=False)
    return fig


# ── Bubble Chart ──────────────────────────────────────────────────────────────
def plot_bubble(df: pd.DataFrame, country: str,
                bubble_col: str = "RH") -> go.Figure:
    """GHI vs Tamb bubble chart; bubble size = RH or BP."""
    cols = ["GHI", "Tamb", bubble_col]
    sample = df[[c for c in cols if c in df.columns]].dropna()
    sample = sample.sample(min(1500, len(sample)), random_state=7)
    size_norm = (sample[bubble_col] - sample[bubble_col].min()) / \
                (sample[bubble_col].max() - sample[bubble_col].min()) * 20 + 3
    fig = go.Figure(go.Scatter(
        x=sample["Tamb"], y=sample["GHI"],
        mode="markers",
        marker=dict(
            size=size_norm,
            color=sample[bubble_col],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=f"{bubble_col}"),
            opacity=0.6,
        ),
        text=sample[bubble_col].round(1),
    ))
    fig.update_layout(
        title=f"{country} — GHI vs Tamb (bubble = {bubble_col})",
        xaxis_title="Ambient Temperature (°C)",
        yaxis_title="GHI (W/m²)",
        template="plotly_white",
        height=480,
    )
    return fig


# ── Cross-Country Comparison ──────────────────────────────────────────────────
def plot_boxplots_comparison(frames: dict, metric: str) -> go.Figure:
    """Side-by-side boxplots for one irradiance metric across all countries."""
    fig = go.Figure()
    for country, df in frames.items():
        if metric in df.columns:
            fig.add_trace(go.Box(
                y=df[metric].dropna(),
                name=country,
                marker_color=COUNTRY_COLORS.get(country, "#888"),
                boxmean="sd",
            ))
    fig.update_layout(
        title=f"{metric} Distribution by Country",
        yaxis_title=f"{metric} (W/m²)",
        template="plotly_white",
        height=480,
    )
    return fig


def plot_avg_ghi_bar(frames: dict) -> go.Figure:
    """Bar chart ranking countries by average GHI."""
    data = {c: df["GHI"].mean() for c, df in frames.items() if "GHI" in df.columns}
    data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
    fig  = go.Figure(go.Bar(
        x=list(data.keys()),
        y=list(data.values()),
        marker_color=[COUNTRY_COLORS.get(c, "#888") for c in data],
        text=[f"{v:.1f}" for v in data.values()],
        textposition="outside",
    ))
    fig.update_layout(
        title="Country Ranking by Average GHI",
        yaxis_title="Average GHI (W/m²)",
        template="plotly_white",
        height=400,
    )
    return fig


def run_anova(frames: dict, metric: str = "GHI") -> dict:
    """
    One-way ANOVA (or Kruskal-Wallis if normality assumption is suspect).
    Returns dict with f_stat, p_value, kruskal_stat, kruskal_p.
    """
    groups = [df[metric].dropna().values
              for df in frames.values() if metric in df.columns]
    if len(groups) < 2:
        return {}
    f_stat,   p_anova   = stats.f_oneway(*groups)
    h_stat,   p_kruskal = stats.kruskal(*groups)
    return {
        "ANOVA F-statistic":   round(f_stat, 4),
        "ANOVA p-value":       round(p_anova, 6),
        "Kruskal-Wallis H":    round(h_stat, 4),
        "Kruskal-Wallis p":    round(p_kruskal, 6),
    }
