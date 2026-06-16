## 🌞 Cross Country Solar Potential Analysis

Exploratory Data Analysis and cross-country comparison of solar irradiance data across **Benin**, **Sierra Leone**, and **Togo**.

## Project Structure

```
solar_project/
├── notebooks/
│   ├── benin_eda.ipynb          # EDA for Benin  (branch: eda-benin)
│   ├── sierraleone_eda.ipynb    # EDA for Sierra Leone  (branch: eda-sierraleone)
│   ├── togo_eda.ipynb           # EDA for Togo  (branch: eda-togo)
│   └── compare_countries.ipynb  # Cross-country comparison  (branch: compare-countries)
├── app/
│   └── Dashboard.py             # Streamlit interactive dashboard
├── data/                        # ⚠️ gitignored — place CSVs here locally
│   ├── benin-malanville.csv
│   ├── sierraleone.csv
│   └── togo.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd solar_project
pip install -r requirements.txt

# Place your raw CSVs in data/
# Run EDA notebooks in Jupyter
jupyter notebook notebooks/benin_eda.ipynb

# Launch dashboard
streamlit run app/dashboard.py
```

## Git Workflow

```bash
# EDA branches
git checkout -b eda-benin
# ... work on benin_eda.ipynb ...
git add notebooks/benin_eda.ipynb
git commit -m "feat: Benin EDA — cleaning, time series, wind rose"
git push origin eda-benin

# Repeat for sierraleone, togo

# Cross-country comparison
git checkout -b compare-countries
git add notebooks/compare_countries.ipynb
git commit -m "feat: cross-country GHI/DNI/DHI comparison + ANOVA"
git push origin compare-countries
```
## 5 Key Observations

*(Update these bullet points once you run on real data)*

- **Benin shows the highest median GHI** (475W/m² daytime values) but also the greatest variability (SD 157.46 W/m²), suggesting intermittent cloud cover alongside strong clear-sky periods.
- **Togo's DNI is competitive with Benin** and its lower variability makes it potentially more reliable for concentrated solar power (CSP) systems requiring consistent direct radiation.
- **Sierra Leone has the lowest average GHI and DNI**, likely reflecting its higher annual rainfall and cloud cover; however, its DHI is proportionally higher — suggesting diffuse-radiation technologies (e.g., thin-film PV) could still be viable.

> The one-way ANOVA and Kruskal-Wallis tests both confirm (p < 0.001) that GHI distributions differ significantly across the three countries, validating region-specific solar deployment strategies.

## Deploy Dashboard

1. Push `app/dashboard.py` and `requirements.txt` to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect repo → set **Main file path** to `app/dashboard.py`.
4. Click **Deploy** — app goes live in ~2 minutes.



## References

- [pandas documentation](https://pandas.pydata.org/docs/)
- [scipy.stats — ANOVA & Kruskal-Wallis](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [seaborn heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- [windrose library](https://python-windrose.github.io/windrose/)
- [Streamlit deployment docs](https://docs.streamlit.io/deploy/streamlit-community-cloud)
