# Solar Investment Strategy Analysis

**Analytics Engineer**

## Project Overview

This project performs end-to-end exploratory data analysis (EDA) on solar radiation measurement data across **Benin**, **Sierra Leone**, and **Togo** to identify high-potential regions for solar installation that align with MoonLight Energy Solutions' long-term sustainability goals.

---

## Repository Structure

```
moonlight-solar/
├── app/
│   ├── __init__.py
│   ├── main.py          # Streamlit dashboard entry point
│   └── utils.py         # Data processing & visualization utilities
├── scripts/
│   ├── __init__.py
│   └── README.md
├── notebooks/
│   ├── benin_eda.ipynb
│   ├── sierraleone_eda.ipynb
│   ├── togo_eda.ipynb
│   └── compare_countries.ipynb
├── data/                # ← gitignored, never committed
│   ├── benin_clean.csv
│   ├── sierraleone_clean.csv
│   └── togo_clean.csv
├── reports/
│   └── strategy_report.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/moonlight-solar.git
cd moonlight-solar

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place raw CSVs in data/ (gitignored)
#    data/benin-malanville_qc.csv
#    data/sierraleone-bumbuna_qc.csv
#    data/togo-dapaong_qc.csv

# 5. Run EDA notebooks
jupyter notebook notebooks/

# 6. Launch Streamlit dashboard
streamlit run app/main.py
```

---

## Branches

| Branch | Purpose |
|---|---|
| `main` | Stable, reviewed code |
| `eda-benin` | EDA notebook for Benin |
| `eda-sierraleone` | EDA notebook for Sierra Leone |
| `eda-togo` | EDA notebook for Togo |
| `compare-countries` | Cross-country comparison notebook |
| `dashboard` | Streamlit app |

---

## Dataset Columns

| Column | Unit | Description |
|---|---|---|
| Timestamp | yyyy-mm-dd hh:mm | Date and time of observation |
| GHI | W/m² | Global Horizontal Irradiance |
| DNI | W/m² | Direct Normal Irradiance |
| DHI | W/m² | Diffuse Horizontal Irradiance |
| ModA | W/m² | Sensor module A reading |
| ModB | W/m² | Sensor module B reading |
| Tamb | °C | Ambient temperature |
| RH | % | Relative humidity |
| WS | m/s | Wind speed |
| WSgust | m/s | Wind gust speed |
| WSstdev | m/s | Wind speed standard deviation |
| WD | °N | Wind direction |
| WDstdev | — | Wind direction standard deviation |
| BP | hPa | Barometric pressure |
| Cleaning | 0/1 | Cleaning event flag |
| Precipitation | mm/min | Precipitation rate |
| TModA | °C | Module A temperature |
| TModB | °C | Module B temperature |
| Comments | — | Free-text notes |

---

## References

- [Solar Radiation Measurement Data — energydata.info](https://energydata.info/dataset/?q=Solar+Radiation+Measurement&vocab_regions=AFR)
- Duffie, J.A. & Beckman, W.A. (2013). *Solar Engineering of Thermal Processes*. Wiley.
- [Windrose Python library](https://python-windrose.github.io/windrose/)
- [Seaborn statistical visualization](https://seaborn.pydata.org/)
- [Streamlit documentation](https://docs.streamlit.io/)
