"""
app/main.py
-----------
MoonLight Energy Solutions — Solar Investment Strategy Dashboard

Run:
    streamlit run app/main.py
"""

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from app.utils import (
    load_all,
    summary_table,
    plot_time_series,
    plot_monthly_heatmap,
    plot_cleaning_impact,
    plot_correlation_heatmap,
    plot_scatter_wind_ghi,
    plot_rh_vs_tamb,
    plot_wind_rose_plotly,
    plot_histograms,
    plot_bubble,
    plot_boxplots_comparison,
    plot_avg_ghi_bar,
    run_anova,
    IRRADIANCE_COLS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="solar Energy Analysis",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0d1b2a; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .main-title {
        font-size: 2rem; font-weight: 700;
        color: #1F4E8C; margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem; color: #555; margin-top: 0;
    }
    .kpi-card {
        background: #f0f4ff; border-radius: 10px;
        padding: 16px 20px; text-align: center;
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1F4E8C; }
    .kpi-label { font-size: 0.85rem; color: #666; }
    .insight-box {
        background: #fffbe6; border-left: 4px solid #f4a261;
        padding: 12px 16px; border-radius: 6px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading cleaned datasets…")
def get_data():
    # Support running from repo root or from app/
    for candidate in ["data", "../data", os.path.join(os.path.dirname(__file__), "../data")]:
        frames = load_all(candidate)
        if frames:
            return frames
    return {}


frames = get_data()
available = list(frames.keys())


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/sun.png", width=64)
    st.markdown("## ☀️ MoonLight Energy")
    st.markdown("**Solar Investment Analysis**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 Country EDA",
         "🌍 Cross-Country Comparison",
         "📋 Strategy Report"],
    )

    st.markdown("---")
    if available:
        selected_country = st.selectbox("Select Country (EDA)", available)
    else:
        selected_country = None

    resample_opts = {"Hourly": "h", "Daily": "D", "Weekly": "W", "Monthly": "ME"}
    resample_label = st.selectbox("Time Aggregation", list(resample_opts.keys()), index=1)
    resample = resample_opts[resample_label]

    bubble_col = st.selectbox("Bubble Chart Size", ["RH", "BP"], index=0)

    st.markdown("---")
    st.caption("Data: energydata.info | Solar Radiation Measurement")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-title">MoonLight Energy Solutions</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Solar Investment Strategy — Environmental Data Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not frames:
        st.warning("""
        **No cleaned data found in `data/` directory.**

        Please run the cleaning pipeline first:
        ```bash
        python scripts/clean_data.py --country benin --input data/benin-malanville_qc.csv
        python scripts/clean_data.py --country sierraleone --input data/sierraleone-bumbuna_qc.csv
        python scripts/clean_data.py --country togo --input data/togo-dapaong_qc.csv
        ```
        Then refresh this page.
        """)
        st.stop()

    # KPI cards
    st.subheader("Key Metrics at a Glance")
    cols = st.columns(len(frames) * 2)
    idx  = 0
    for country, df in frames.items():
        with cols[idx]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{df["GHI"].mean():.0f}</div>
                <div class="kpi-label">{country}<br>Avg GHI (W/m²)</div>
            </div>""", unsafe_allow_html=True)
        idx += 1
        with cols[idx]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{df["GHI"].max():.0f}</div>
                <div class="kpi-label">{country}<br>Peak GHI (W/m²)</div>
            </div>""", unsafe_allow_html=True)
        idx += 1

    st.markdown("---")

    # Country ranking bar
    st.subheader("Country Ranking by Average GHI")
    st.plotly_chart(plot_avg_ghi_bar(frames), use_container_width=True)

    # Summary table
    st.subheader("Cross-Country Summary Statistics")
    tbl = summary_table(frames)
    st.dataframe(
        tbl.style.background_gradient(subset=["Mean","Median"], cmap="YlOrRd"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("""
    <div class="insight-box">
    <b>How to use this dashboard:</b><br>
    Use the sidebar to navigate between pages. <b>Country EDA</b> gives deep-dive
    analysis per country. <b>Cross-Country Comparison</b> synthesises all three
    datasets with statistical testing. <b>Strategy Report</b> presents the final
    investment recommendation.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: COUNTRY EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Country EDA":
    if not frames or selected_country not in frames:
        st.warning("No data loaded. See Overview page for setup instructions.")
        st.stop()

    df = frames[selected_country]

    st.markdown(f'<p class="main-title">{selected_country} — Exploratory Data Analysis</p>',
                unsafe_allow_html=True)
    st.markdown(f"**{len(df):,} observations** after cleaning  |  "
                f"Period: {df.index.min().date()} → {df.index.max().date()}")
    st.markdown("---")

    # ── Tab layout ────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Time Series",
        "🧹 Cleaning Impact",
        "🔗 Correlations",
        "💨 Wind & Distributions",
        "🌡️ Temperature & RH",
        "🫧 Bubble Chart",
        "📋 Raw Stats",
    ])

    # Tab 1 — Time Series
    with tabs[0]:
        st.subheader("Solar Irradiance & Temperature Over Time")
        st.plotly_chart(plot_time_series(df, selected_country, resample),
                        use_container_width=True)
        st.subheader("Hour × Month Heatmap")
        metric_hm = st.selectbox("Select metric", ["GHI", "DNI", "DHI", "Tamb"],
                                 key="hm_metric")
        st.plotly_chart(plot_monthly_heatmap(df, metric_hm), use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        <b>What to look for:</b> GHI peaks mid-day (10:00–14:00) and is highest
        in dry-season months. Overnight and pre-dawn values should be ~0 W/m²;
        anomalies here indicate sensor faults.
        </div>
        """, unsafe_allow_html=True)

    # Tab 2 — Cleaning Impact
    with tabs[1]:
        st.subheader("Sensor Output: Pre vs Post Cleaning")
        st.plotly_chart(plot_cleaning_impact(df, selected_country),
                        use_container_width=True)
        if "Cleaning" in df.columns:
            grp = df.groupby("Cleaning")[["ModA","ModB"]].mean()
            grp.index = ["Not Cleaned","Cleaned"]
            st.dataframe(grp.round(2))
        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b> A meaningful lift in ModA/ModB readings after cleaning
        events confirms that dust/soiling is a significant efficiency loss factor
        in this region and supports regular maintenance scheduling.
        </div>
        """, unsafe_allow_html=True)

    # Tab 3 — Correlations
    with tabs[2]:
        st.subheader("Correlation Heatmap")
        fig_corr = plot_correlation_heatmap(df, selected_country)
        st.pyplot(fig_corr)
        plt.close()

        st.subheader("Wind Speed vs GHI")
        st.plotly_chart(plot_scatter_wind_ghi(df, selected_country),
                        use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b> Strong GHI–DNI correlation confirms direct irradiance
        dominates on clear days. Negative GHI–RH correlation supports that
        cloud cover (high humidity) suppresses solar radiation.
        </div>
        """, unsafe_allow_html=True)

    # Tab 4 — Wind & Distributions
    with tabs[3]:
        st.subheader("Wind Rose")
        st.plotly_chart(plot_wind_rose_plotly(df, selected_country),
                        use_container_width=True)
        st.subheader("GHI & Wind Speed Distributions")
        hist_cols = [c for c in ["GHI", "WS"] if c in df.columns]
        st.plotly_chart(plot_histograms(df, selected_country, hist_cols),
                        use_container_width=True)

    # Tab 5 — Temperature & RH
    with tabs[4]:
        st.subheader("Relative Humidity vs Ambient Temperature")
        st.plotly_chart(plot_rh_vs_tamb(df, selected_country),
                        use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b> High relative humidity tends to coincide with cooler
        ambient temperatures (rainy/cloud season). High humidity also depresses
        GHI — shown by colour gradient shifting to blue/purple in humid clusters.
        </div>
        """, unsafe_allow_html=True)

    # Tab 6 — Bubble Chart
    with tabs[5]:
        st.subheader(f"GHI vs Tamb (bubble = {bubble_col})")
        st.plotly_chart(plot_bubble(df, selected_country, bubble_col),
                        use_container_width=True)
        st.caption(
            f"Each point = one observation. Bubble size and colour encode {bubble_col}. "
            "Larger bubbles in high-GHI / high-Tamb zone indicate optimal solar conditions."
        )

    # Tab 7 — Raw Stats
    with tabs[6]:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().T.round(3), use_container_width=True)
        st.subheader("Missing Values")
        null_df = df.isna().sum().rename("Null Count").to_frame()
        null_df["Null %"] = (null_df["Null Count"] / len(df) * 100).round(2)
        st.dataframe(null_df[null_df["Null Count"] > 0], use_container_width=True)
        if null_df["Null Count"].sum() == 0:
            st.success("✅ No missing values in the cleaned dataset.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CROSS-COUNTRY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Cross-Country Comparison":
    st.markdown('<p class="main-title">Cross-Country Solar Potential Comparison</p>',
                unsafe_allow_html=True)
    st.markdown("Benin · Sierra Leone · Togo")
    st.markdown("---")

    if len(frames) < 2:
        st.warning("Need at least 2 country datasets to compare. See Overview for setup.")
        st.stop()

    # Boxplots
    st.subheader("Irradiance Distributions by Country")
    col1, col2, col3 = st.columns(3)
    for col_widget, metric in zip([col1, col2, col3], ["GHI", "DNI", "DHI"]):
        with col_widget:
            st.plotly_chart(plot_boxplots_comparison(frames, metric),
                            use_container_width=True)

    # Summary table
    st.subheader("Statistical Summary Table")
    tbl = summary_table(frames)
    st.dataframe(
        tbl.style.background_gradient(subset=["Mean","Median"], cmap="YlOrRd"),
        use_container_width=True,
        hide_index=True,
    )

    # Ranking bar
    st.subheader("Country Ranking — Average GHI")
    st.plotly_chart(plot_avg_ghi_bar(frames), use_container_width=True)

    # Statistical tests
    st.subheader("Statistical Testing")
    test_metric = st.selectbox("Test metric", ["GHI", "DNI", "DHI"], key="test_m")
    results     = run_anova(frames, test_metric)
    if results:
        r1, r2 = st.columns(2)
        with r1:
            st.metric("ANOVA F-statistic", results["ANOVA F-statistic"])
            st.metric("ANOVA p-value",     results["ANOVA p-value"])
        with r2:
            st.metric("Kruskal-Wallis H",  results["Kruskal-Wallis H"])
            st.metric("Kruskal-Wallis p",  results["Kruskal-Wallis p"])
        p = results["ANOVA p-value"]
        if p < 0.05:
            st.success(
                f"✅ p = {p:.6f} < 0.05 — The difference in {test_metric} across "
                "countries is **statistically significant**. Country choice materially "
                "affects expected solar yield."
            )
        else:
            st.info(f"p = {p:.6f} ≥ 0.05 — No statistically significant difference detected.")

    # Key observations
    st.subheader("Key Observations")
    st.markdown("""
    <div class="insight-box">
    🔹 <b>Observation 1 — GHI Magnitude:</b>
    Benin (Malanville) consistently records the highest median GHI, reflecting its
    Sahelian location at lower latitude with drier, clearer skies and fewer cloud-cover days.
    </div>
    <div class="insight-box">
    🔹 <b>Observation 2 — Variability:</b>
    Sierra Leone (Bumbuna) shows the greatest GHI variability (widest IQR),
    driven by its humid tropical climate and pronounced wet season. High variability
    increases storage/backup requirements for any solar installation.
    </div>
    <div class="insight-box">
    🔹 <b>Observation 3 — Stability vs Peak:</b>
    Togo (Dapaong) offers a balance — competitive median GHI with lower
    variability than Sierra Leone. This stable irradiance profile reduces intermittency
    risk, making it attractive for utility-scale installations with firm capacity targets.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: STRATEGY REPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Strategy Report":
    st.markdown('<p class="main-title">Strategic Investment Recommendation</p>',
                unsafe_allow_html=True)
    st.markdown("*MoonLight Energy Solutions — Solar Installation Strategy Report*")
    st.markdown("---")

    st.markdown("""
    ## Executive Summary

    MoonLight Energy Solutions seeks to identify high-potential regions for solar
    installation that align with long-term sustainability and operational efficiency goals.
    Based on statistical analysis of solar radiation measurement data from **Benin**,
    **Sierra Leone**, and **Togo**, this report presents a data-driven investment strategy.

    ---

    ## Methodology

    | Step | Action |
    |---|---|
    | Data Profiling | Summary statistics, null audit (>5% flag), dtype checks |
    | Cleaning | Negative irradiance clamping, median imputation, Z-score outlier removal (|Z|>3) |
    | EDA | Time-series, heatmaps, correlation matrices, wind roses, bubble charts |
    | Statistical Testing | One-way ANOVA + Kruskal-Wallis on GHI across countries |

    ---

    ## Key Findings

    ### 1. Solar Resource Quality
    - **Benin (Malanville)** records the **highest average GHI** among the three sites,
      consistent with its arid Sahelian climate (low cloud cover, minimal precipitation).
    - **Togo (Dapaong)** achieves competitive irradiance with notably **lower variability**,
      reducing intermittency risk for utility-scale projects.
    - **Sierra Leone (Bumbuna)** has the **most variable solar resource**, driven by
      its humid tropical climate and long wet season — requiring larger battery/storage buffer.

    ### 2. Cleaning & Maintenance
    Sensor readings (ModA, ModB) show a statistically meaningful increase post-cleaning
    events in all three countries, confirming that **dust and soiling are significant
    efficiency loss factors**. A regular cleaning schedule (aligned with dry-season dust
    peaks) is recommended for any installation.

    ### 3. Temperature & Humidity
    - Higher ambient temperatures in the 35–40 °C range suppress module efficiency
      (thermal coefficient losses). This is most pronounced in Benin.
    - High RH periods coincide with reduced GHI — scheduling maintenance during
      these windows minimises lost generation.

    ### 4. Wind Conditions
    Wind speeds across all sites are mild (2–5 m/s typical), posing minimal structural
    risk to standard utility-scale PV racking systems.

    ---

    ## Strategic Recommendation

    | Priority | Country | Rationale |
    |---|---|---|
    | 🥇 **Primary** | **Benin (Malanville)** | Highest average & peak GHI; lowest cloud frequency; strongest return on kWh/kWp |
    | 🥈 **Secondary** | **Togo (Dapaong)** | Second-best GHI with lowest variability; lower storage CAPEX required |
    | 🥉 **Conditional** | **Sierra Leone (Bumbuna)** | Viable if paired with adequate battery storage; consider hybrid solar-hydro given water availability |

    ### Implementation Priorities
    1. **Site Feasibility Study** — Ground-truth the Malanville and Dapaong sites with
       12-month meteorological stations before committing capital.
    2. **Cleaning Protocol** — Implement automated soiling monitors and schedule cleaning
       every 2–4 weeks during dry season.
    3. **Storage Sizing** — For Sierra Leone, use the measured GHI standard deviation
       to right-size battery storage (recommend ≥4 hours at peak load).
    4. **Phased Rollout** — Pilot 5 MW in Benin (Year 1), expand to Togo (Year 2),
       evaluate Sierra Leone hybrid model (Year 3).

    ---

    ## References

    - energydata.info — Solar Radiation Measurement Data (West Africa)
    - Duffie, J.A. & Beckman, W.A. (2013). *Solar Engineering of Thermal Processes*. Wiley.
    - IRENA (2023). *Renewable Power Generation Costs in Africa*.
    - [Windrose Python library](https://python-windrose.github.io/windrose/)
    - SciPy one-way ANOVA: `scipy.stats.f_oneway`
    """)

    if frames:
        st.markdown("---")
        st.subheader("Live Data: Country Ranking")
        st.plotly_chart(plot_avg_ghi_bar(frames), use_container_width=True)
        st.subheader("Statistical Significance — GHI Differences")
        results = run_anova(frames, "GHI")
        if results:
            st.json(results)
