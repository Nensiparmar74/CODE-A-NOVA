# CODE-A-NOVA Month 3 — COVID-19 Data Analysis Dashboard

## Project overview
An interactive dashboard for exploring COVID-19 cases, deaths, vaccination coverage, and time-series trends. It supports country/region filters, date-range selection, summary metrics, interactive Plotly charts, a summary table, and CSV export.

This project maps to the **Level 2(b): COVID-19 Data Analysis Dashboard** task in the CODE-A-NOVA Month 3 Data Science brief, which asks for:
- Data cleaning and transformation
- Time-series analysis
- Interactive visualizations (Plotly / Power BI / Tableau)
- A dashboard showing trends and statistics

## Data source
The dashboard uses the Our World in Data COVID-19 dataset:
- Dataset information: https://ourworldindata.org/coronavirus
- Data explorer: https://ourworldindata.org/explorers/coronavirus-data-explorer
- Downloadable data: https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv

The dashboard loads the dataset from the source URL when `data/owid-covid-data.csv` is not present. If you prefer a local copy, download a CSV from the OWID data source and save it as `data/owid-covid-data.csv`. The dataset is not bundled in this ZIP because it is large and changes over time.

## Features
- Country/region selector
- Date-range filter
- Summary metrics for latest available cases and deaths
- Time-series charts for reported cases and deaths
- Vaccination coverage chart when available
- Latest available data table
- Download filtered data as CSV
- Caching for faster subsequent loads

## Tech stack
- Python
- Pandas
- Streamlit
- Plotly Express

## Setup (Windows)
1. Install Python 3.9 or later.
2. Extract this ZIP.
3. Open PowerShell in the extracted project folder.
4. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

5. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

6. Start the dashboard:

```powershell
streamlit run app.py
```

The app will open in your browser. An internet connection is required if you do not have a local dataset.

## Project structure

```text
CODE-A-NOVA-COVID19-Dashboard/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── GITHUB_UPLOAD.txt
├── data/
│   └── README.txt
└── outputs/
    └── README.txt
```

## Data processing and analysis
The app parses dates, checks for essential columns, selects locations and a date range, handles missing chart values by excluding them from the relevant chart, and presents time-series trends. It uses the available columns in the downloaded dataset version.

## Important interpretation note
Reported counts are affected by testing availability, reporting practices, and later revisions. When several locations are selected, the summary metrics add the latest row from each selected location; selecting both a parent region and its subregions can double-count. Use the chart/table to understand the selected locations and date range.

## Limitations
- This is an exploratory dashboard, not a clinical or forecasting model.
- Data coverage and column availability can change over time.
- A missing value does not necessarily mean zero cases or deaths.
- Figures should be interpreted in the context of reporting limitations.

## GitHub upload
See `GITHUB_UPLOAD.txt` for the commands to create a repository and push this project.
