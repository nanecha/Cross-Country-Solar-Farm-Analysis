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

## Key Findings (update after running on real data)

- **Benin** shows highest median GHI but greatest variability.
- **Togo** has competitive DNI with lower variance — good for CSP systems.
- **Sierra Leone** has lowest GHI; higher DHI-to-GHI ratio favours diffuse-radiation PV.
- ANOVA confirms statistically significant GHI differences across countries (p < 0.001).

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
