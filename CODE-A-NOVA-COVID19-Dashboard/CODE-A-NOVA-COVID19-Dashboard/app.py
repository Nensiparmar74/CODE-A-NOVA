"""
CODE-A-NOVA Month 3 Level 2(b)
COVID-19 Data Analysis Dashboard

Run:
    python -m pip install -r requirements.txt
    streamlit run app.py

Data source:
    Our World in Data COVID-19 dataset (download URL is also in README.md).
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
LOCAL_CSV = DATA_DIR / "owid-covid-data.csv"
DATA_URL = (
    "https://catalog.ourworldindata.org/garden/covid/latest/"
    "compact/compact.csv"
)

st.set_page_config(
    page_title="COVID-19 Data Analysis Dashboard",
    page_icon="🦠",
    layout="wide",
)

st.title("COVID-19 Data Analysis Dashboard")
st.caption("CODE-A-NOVA · Month 3 · Level 2(b)")
st.markdown(
    "Explore reported COVID-19 cases, deaths, vaccinations, and trends "
    "using country-level time-series data."
)

@st.cache_data(show_spinner=False)
def load_data():
    if LOCAL_CSV.exists():
        data = pd.read_csv(LOCAL_CSV, low_memory=False)
        source_note = f"Local dataset: {LOCAL_CSV.name}"
    else:
        data = pd.read_csv(DATA_URL, low_memory=False)
        source_note = "Source: Our World in Data (loaded online)"
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data, source_note

try:
    with st.spinner("Loading COVID-19 dataset..."):
        df, source_note = load_data()
except Exception as exc:
    st.error("Could not load the COVID-19 dataset.")
    st.write("Check your internet connection, or download the CSV described in README.md into the data folder.")
    st.exception(exc)
    st.stop()

required = {"location", "date"}
missing_required = required - set(df.columns)
if missing_required:
    st.error(f"Dataset is missing required columns: {', '.join(sorted(missing_required))}")
    st.stop()

st.caption(source_note)

# Select country/region and date range.
locations = sorted(df["location"].dropna().unique().tolist())
default_countries = [c for c in ["World", "India", "United States"] if c in locations]
if not default_countries:
    default_countries = locations[:1]

with st.sidebar:
    st.header("Dashboard filters")
    selected = st.multiselect(
        "Select countries / regions",
        options=locations,
        default=default_countries,
    )
    valid_dates = df["date"].dropna()
    if valid_dates.empty:
        st.error("No valid dates were found in the dataset.")
        st.stop()
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    st.caption("Reported figures may be revised and may be incomplete for some locations.")

if not selected:
    st.info("Select at least one country or region in the sidebar.")
    st.stop()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = date_range

filtered = df[
    df["location"].isin(selected)
    & df["date"].dt.date.between(start_date, end_date)
].copy()

def numeric_column(frame, candidates):
    for name in candidates:
        if name in frame.columns:
            return name
    return None

cases_col = numeric_column(filtered, ["new_cases", "new_cases_smoothed"])
deaths_col = numeric_column(filtered, ["new_deaths", "new_deaths_smoothed"])
vacc_col = numeric_column(filtered, ["people_vaccinated_per_hundred", "people_fully_vaccinated_per_hundred"])

latest = (
    filtered.sort_values("date")
    .groupby("location", as_index=False)
    .tail(1)
)

def sum_if_exists(frame, column):
    if column and column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(0).sum()
    return None

cases_total = sum_if_exists(latest, "total_cases")
deaths_total = sum_if_exists(latest, "total_deaths")
if cases_total is None:
    cases_total = sum_if_exists(latest, "total_cases_per_million")
if deaths_total is None:
    deaths_total = sum_if_exists(latest, "total_deaths_per_million")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest reported total cases*", f"{cases_total:,.0f}" if cases_total is not None else "N/A")
c2.metric("Latest reported total deaths*", f"{deaths_total:,.0f}" if deaths_total is not None else "N/A")
c3.metric("Selected locations", f"{len(selected):,}")
c4.metric("Dates in selection", f"{filtered['date'].nunique():,}")
st.caption("*Totals sum the latest available row for each selected location; they are not necessarily global totals and may double-count regions when both a parent region and its subregions are selected.")

st.subheader("Reported cases over time")
if cases_col:
    cases_plot = filtered.dropna(subset=[cases_col])
    fig = px.line(
        cases_plot, x="date", y=cases_col, color="location",
        title=f"{cases_col.replace('_', ' ').title()}",
        labels={"date": "Date", cases_col: "Reported cases"},
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Cases time-series column is unavailable in this dataset version.")

st.subheader("Reported deaths over time")
if deaths_col:
    deaths_plot = filtered.dropna(subset=[deaths_col])
    fig = px.line(
        deaths_plot, x="date", y=deaths_col, color="location",
        title=f"{deaths_col.replace('_', ' ').title()}",
        labels={"date": "Date", deaths_col: "Reported deaths"},
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Deaths time-series column is unavailable in this dataset version.")

st.subheader("Vaccination coverage")
if vacc_col:
    vacc_plot = filtered.dropna(subset=[vacc_col])
    fig = px.line(
        vacc_plot, x="date", y=vacc_col, color="location",
        title=f"{vacc_col.replace('_', ' ').title()}",
        labels={"date": "Date", vacc_col: "People per hundred"},
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Vaccination coverage is not available in the selected dataset columns.")

st.subheader("Latest available country / region summary")
summary_cols = [c for c in [
    "location", "date", "total_cases", "total_deaths",
    "new_cases", "new_deaths", "people_vaccinated_per_hundred",
    "people_fully_vaccinated_per_hundred"
] if c in latest.columns]
st.dataframe(
    latest[summary_cols].sort_values("date", ascending=False),
    use_container_width=True,
    hide_index=True,
)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered data as CSV",
    data=csv_bytes,
    file_name="covid19_filtered_data.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption("Data interpretation: reported counts depend on testing, reporting practices, and later revisions. They should not be interpreted as a complete measure of infections or mortality.")
