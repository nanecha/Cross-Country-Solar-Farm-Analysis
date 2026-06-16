"""
Solar Potential Dashboard — Benin · Sierra Leone · Togo
Run locally:  streamlit run app/dashboard.py
Deploy:       https://docs.streamlit.io/deploy/streamlit-community-cloud
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats
from pathlib import Path

import streamlit as st

# ── Try optional heavy imports ───────────────────────────────────
try:
    from windrose import WindroseAxes
    HAS_WINDROSE = True
except ImportError:
    HAS_WINDROSE = False

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="West Africa Solar Dashboard",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styling ────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: linear-gradient(135deg, #1E3A5F, #2980B9);
    border-radius: 10px; padding: 16px 20px; color: white;
    text-align: center; margin: 4px;
  }
  .metric-card h3 { font-size: 1.8rem; margin: 0; }
  .metric-card p  { margin: 0; opacity: 0.85; font-size: 0.85rem; }
  .section-header { color: #1E3A5F; border-bottom: 2px solid #2980B9; padding-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

PALETTE = {"Benin": "#E74C3C", "Sierra Leone": "#2980B9", "Togo": "#27AE60"}
METRICS = ["GHI", "DNI", "DHI"]
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# ════════════════════════════════════════════════════════════════
#  DATA LOADING  (reads cleaned CSVs or generates synthetic data
# ════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading datasets…")
def load_data() -> dict[str, pd.DataFrame]:
    DATA_DIR = Path("data")
    map_ = {
        "Benin":       DATA_DIR / "benin_clean.csv",
        "Sierra Leone": DATA_DIR / "sierraleone_clean.csv",
        "Togo":        DATA_DIR / "Togo_clean.csv",
    }
    frames = {}
    for name, path in map_.items():
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Timestamp"])
        else:
            # synthetic demo
            np.random.seed(list(map_.keys()).index(name) * 17)
            n = 8760
            ts = pd.date_range("2021-01-01", periods=n, freq="h")
            hour = ts.hour
            base = np.clip(np.sin((hour - 6) * np.pi / 12), 0, 1)
            scale = {"Benin": 1.0, "Sierra Leone": 0.88, "Togo": 0.95}[name]
            df = pd.DataFrame({
                "Timestamp":   ts,
                "GHI":  (base * 900 * scale + np.random.normal(0, 45, n)).clip(0),
                "DNI":  (base * 700 * scale + np.random.normal(0, 55, n)).clip(0),
                "DHI":  (base * 200 * scale + np.random.normal(0, 22, n)).clip(0),
                "ModA": 25 + base * 20 + np.random.normal(0, 2, n),
                "ModB": 25 + base * 18 + np.random.normal(0, 2, n),
                "TModA": 25 + base * 22 + np.random.normal(0, 2, n),
                "TModB": 25 + base * 21 + np.random.normal(0, 2, n),
                "Tamb": 28 + base * 8  + np.random.normal(0, 2, n),
                "RH":   np.clip(70 - base * 20 + np.random.normal(0, 5, n), 10, 100),
                "WS":   np.abs(np.random.normal(3, 1.5, n)),
                "WSgust": np.abs(np.random.normal(5, 2, n)),
                "WD":   np.random.uniform(0, 360, n),
                "BP":   np.random.normal(1013, 3, n),
                "Cleaning": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            })
        df["country"] = name
        df["month"]   = pd.to_datetime(df["Timestamp"]).dt.month
        df["hour"]    = pd.to_datetime(df["Timestamp"]).dt.hour
        frames[name] = df
    return frames


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
st.sidebar.image("https://em-content.zobj.net/source/twitter/376/sun_2600-fe0f.png", width=60)
st.sidebar.title("Solar Dashboard")
st.sidebar.markdown("West Africa · 2021")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📈 EDA — Single Country", "🌍 Cross-Country", "💨 Wind Analysis"],
)

all_frames = load_data()
country_names = list(all_frames.keys())
df_all = pd.concat(all_frames.values(), ignore_index=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
selected_countries = st.sidebar.multiselect(
    "Countries", country_names, default=country_names
)
hour_range = st.sidebar.slider("Hour of day", 0, 23, (6, 18))
month_range = st.sidebar.slider("Months", 1, 12, (1, 12))

# Apply filters
mask = (
    df_all["country"].isin(selected_countries) &
    df_all["hour"].between(*hour_range) &
    df_all["month"].between(*month_range)
)
df_filtered = df_all[mask].copy()

st.sidebar.markdown("---")
st.sidebar.caption("Data: Synthetic demo (replace with real CSVs in data/)")


# ════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🌞 West Africa Solar Potential Dashboard")
    st.markdown("Comparative analysis of solar irradiance across **Benin**, **Sierra Leone**, and **Togo**.")

    # KPI row
    cols = st.columns(len(selected_countries) * 3 or 3)
    col_idx = 0
    for country in selected_countries:
        sub = df_filtered[df_filtered["country"] == country]
        for metric in METRICS:
            with cols[col_idx]:
                val = sub[metric].mean()
                st.markdown(f"""
                <div class="metric-card">
                  <p>{country} · {metric}</p>
                  <h3>{val:.0f}</h3>
                  <p>W/m² mean</p>
                </div>""", unsafe_allow_html=True)
            col_idx += 1

    st.markdown("---")

    # Monthly GHI
    st.markdown('<h3 class="section-header">Monthly Average GHI</h3>', unsafe_allow_html=True)
    monthly = df_filtered.groupby(["month", "country"])["GHI"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    for country in selected_countries:
        sub = monthly[monthly["country"] == country]
        ax.plot(sub["month"], sub["GHI"], marker="o", color=PALETTE[country], linewidth=2,
                markersize=6, label=country)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("Mean GHI (W/m²)")
    ax.set_title("Monthly Average GHI by Country", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Ranking bar
    st.markdown('<h3 class="section-header">Country Ranking by Mean GHI</h3>', unsafe_allow_html=True)
    rank = df_filtered.groupby("country")["GHI"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    bars = ax2.barh(rank.index, rank.values,
                    color=[PALETTE[c] for c in rank.index], edgecolor="white", height=0.5)
    for bar, val in zip(bars, rank.values):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f} W/m²", va="center", fontsize=10, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean GHI (W/m²)")
    ax2.set_title("Country GHI Ranking", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ════════════════════════════════════════════════════════════════
#  PAGE 2 — EDA SINGLE COUNTRY
# ════════════════════════════════════════════════════════════════
elif page == "📈 EDA — Single Country":
    st.title("📈 Single-Country EDA")

    country = st.selectbox("Select country", country_names)
    df_c = df_filtered[df_filtered["country"] == country].copy()

    if df_c.empty:
        st.warning("No data for current filters. Adjust sidebar settings.")
        st.stop()

    # Summary stats
    st.markdown('<h3 class="section-header">Summary Statistics</h3>', unsafe_allow_html=True)
    st.dataframe(df_c[METRICS + ["Tamb", "RH", "WS"]].describe().round(2))

    # Missing value report
    null_rpt = df_c.isna().sum().rename("null_count").to_frame()
    null_rpt["null_pct"] = (null_rpt["null_count"] / len(df_c) * 100).round(2)
    st.markdown("**Missing values:**")
    st.dataframe(null_rpt[null_rpt["null_count"] > 0])

    st.markdown("---")

    # Diurnal profile
    st.markdown('<h3 class="section-header">Diurnal Irradiance Profile</h3>', unsafe_allow_html=True)
    hourly = df_c.groupby("hour")[METRICS].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    for metric, color in zip(METRICS, ["#F39C12", "#E74C3C", "#3498DB"]):
        ax.plot(hourly.index, hourly[metric], marker="o", markersize=3,
                color=color, label=metric, linewidth=2)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean W/m²")
    ax.set_title(f"{country} — Diurnal Profile", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Correlation heatmap
    st.markdown('<h3 class="section-header">Correlation Heatmap</h3>', unsafe_allow_html=True)
    corr_cols = [c for c in ["GHI","DNI","DHI","TModA","TModB","Tamb","RH","WS","ModA","ModB"] if c in df_c.columns]
    corr = df_c[corr_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.4, ax=ax3, annot_kws={"size": 8})
    ax3.set_title(f"{country} — Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Bubble chart
    st.markdown('<h3 class="section-header">Bubble Chart: GHI vs Tamb</h3>', unsafe_allow_html=True)
    if all(c in df_c.columns for c in ["GHI", "Tamb", "RH"]):
        sample = df_c[df_c["GHI"] > 0].sample(min(1500, len(df_c)), random_state=42)
        bubble_col = "BP" if "BP" in sample.columns else "RH"
        bsizes = (sample[bubble_col] - sample[bubble_col].min()) / \
                 (sample[bubble_col].max() - sample[bubble_col].min()) * 150 + 5
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        sc = ax4.scatter(sample["Tamb"], sample["GHI"],
                         c=sample["RH"], cmap="RdYlBu_r",
                         s=bsizes, alpha=0.5, edgecolors="none")
        plt.colorbar(sc, ax=ax4, label="RH (%)")
        ax4.set_xlabel("Tamb (°C)")
        ax4.set_ylabel("GHI (W/m²)")
        ax4.set_title(f"{country} — GHI vs Tamb · colour=RH · size={bubble_col}", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # GHI histogram
    st.markdown('<h3 class="section-header">GHI Distribution</h3>', unsafe_allow_html=True)
    fig5, ax5 = plt.subplots(figsize=(9, 3))
    ghi_data = df_c["GHI"].dropna()
    ax5.hist(ghi_data, bins=50, color=PALETTE[country], alpha=0.75, edgecolor="white")
    ax5.axvline(ghi_data.mean(),   color="black", linestyle="--", label=f"Mean {ghi_data.mean():.0f}")
    ax5.axvline(ghi_data.median(), color="red",   linestyle=":",  label=f"Median {ghi_data.median():.0f}")
    ax5.set_xlabel("GHI (W/m²)")
    ax5.set_ylabel("Frequency")
    ax5.set_title(f"{country} — GHI Histogram  (skew={stats.skew(ghi_data):.2f})", fontweight="bold")
    ax5.legend()
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()


# ════════════════════════════════════════════════════════════════
#  PAGE 3 — CROSS-COUNTRY
# ════════════════════════════════════════════════════════════════
elif page == "🌍 Cross-Country":
    st.title("🌍 Cross-Country Comparison")

    # Box plots
    st.markdown('<h3 class="section-header">GHI · DNI · DHI — Box Plots</h3>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, METRICS):
        sub = df_filtered[df_filtered[metric] > 0]
        sns.boxplot(data=sub, x="country", y=metric,
                    order=[c for c in country_names if c in selected_countries],
                    palette=[PALETTE[c] for c in country_names if c in selected_countries],
                    ax=ax, flierprops=dict(marker=".", markersize=2, alpha=0.3), width=0.55)
        ax.set_title(f"{metric} (W/m²)", fontweight="bold")
        ax.set_xlabel("")
    fig.suptitle("Cross-Country Irradiance Comparison (daytime)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Summary table
    st.markdown('<h3 class="section-header">Summary Statistics Table</h3>', unsafe_allow_html=True)
    rows = []
    for country in selected_countries:
        for metric in METRICS:
            s = df_filtered[df_filtered["country"] == country][metric]
            rows.append({"Country": country, "Metric": metric,
                         "Mean": round(s.mean(), 1), "Median": round(s.median(), 1),
                         "Std Dev": round(s.std(), 1)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Statistical tests
    st.markdown('<h3 class="section-header">Statistical Tests on GHI</h3>', unsafe_allow_html=True)
    groups = [df_filtered[(df_filtered["country"] == c) & (df_filtered["GHI"] > 0)]["GHI"].dropna().values
              for c in selected_countries]
    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
        f_stat, p_anova = stats.f_oneway(*groups)
        h_stat, p_kw    = stats.kruskal(*groups)
        col1, col2 = st.columns(2)
        col1.metric("ANOVA F-statistic", f"{f_stat:.2f}", f"p = {p_anova:.2e}")
        col2.metric("Kruskal-Wallis H",  f"{h_stat:.2f}", f"p = {p_kw:.2e}")
        if p_anova < 0.05:
            st.success("✔ Differences between countries are **statistically significant** (p < 0.05).")
        else:
            st.info("No significant difference at α = 0.05.")
    else:
        st.info("Select at least 2 countries for statistical tests.")

    # Key observations
    st.markdown('<h3 class="section-header">Key Observations</h3>', unsafe_allow_html=True)
    st.markdown("""
- **Benin** shows the highest median GHI but also the greatest variability — intermittent cloud cover alongside strong clear-sky periods.
- **Togo** has competitive DNI with lower variability — better suited for concentrated solar power (CSP) systems.
- **Sierra Leone** has the lowest GHI/DNI, reflecting higher rainfall and cloud cover; proportionally higher DHI suggests diffuse-radiation PV technologies remain viable.
    """)


# ════════════════════════════════════════════════════════════════
#  PAGE 4 — WIND ANALYSIS
# ════════════════════════════════════════════════════════════════
elif page == "💨 Wind Analysis":
    st.title("💨 Wind Analysis")

    country = st.selectbox("Select country", country_names)
    df_c = df_filtered[df_filtered["country"] == country].copy()

    st.markdown('<h3 class="section-header">Wind Speed Distribution</h3>', unsafe_allow_html=True)
    if "WS" in df_c.columns:
        fig, ax = plt.subplots(figsize=(9, 3))
        ws_data = df_c["WS"].dropna()
        ax.hist(ws_data, bins=40, color=PALETTE[country], alpha=0.75, edgecolor="white")
        ax.axvline(ws_data.mean(), color="black", linestyle="--", label=f"Mean {ws_data.mean():.1f} m/s")
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{country} — Wind Speed Histogram", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<h3 class="section-header">Wind Direction Radial Plot</h3>', unsafe_allow_html=True)
    if "WD" in df_c.columns and "WS" in df_c.columns:
        bins = np.arange(0, 361, 22.5)
        labels_wd = [f"{int(b)}°" for b in bins[:-1]]
        df_c["WD_bin"] = pd.cut(df_c["WD"], bins=bins, labels=labels_wd, include_lowest=True)
        wind_agg = df_c.groupby("WD_bin", observed=True)["WS"].mean()
        N = len(wind_agg)
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        fig2, ax2 = plt.subplots(subplot_kw=dict(polar=True), figsize=(7, 7))
        ax2.bar(theta, wind_agg.values, width=2*np.pi/N, alpha=0.75,
                color=cm.viridis(np.linspace(0, 1, N)), edgecolor="white")
        ax2.set_xticks(theta)
        ax2.set_xticklabels(wind_agg.index, fontsize=7)
        ax2.set_title(f"{country} — Mean Wind Speed by Direction", fontweight="bold", pad=20)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown('<h3 class="section-header">WS & WSgust vs GHI</h3>', unsafe_allow_html=True)
    cols_plot = [c for c in ["WS", "WSgust"] if c in df_c.columns]
    if cols_plot:
        fig3, axes3 = plt.subplots(1, len(cols_plot), figsize=(12, 4))
        if len(cols_plot) == 1:
            axes3 = [axes3]
        for ax, col in zip(axes3, cols_plot):
            ax.scatter(df_c[col], df_c["GHI"], alpha=0.1, s=5, color=PALETTE[country])
            r = df_c[[col, "GHI"]].corr().iloc[0, 1]
            ax.set_xlabel(col)
            ax.set_ylabel("GHI (W/m²)")
            ax.set_title(f"{col} vs GHI (r={r:.2f})", fontsize=10)
        fig3.suptitle(f"{country} — Wind vs Solar", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
