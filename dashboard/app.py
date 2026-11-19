"""
Market Segmentation & National Baseline Dashboard
Analyzes German Tech Roles using Entgeldatlas Baselines and FastAPI Clustered Data.
"""

import ast
import json
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pydeck as pdk
import requests
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st

# -----------------------------------------------------------------------------
# DATA SOURCES, CONSTANTS & SCHEMAS
# -----------------------------------------------------------------------------

FASTAPI_BASE_URL = "http://fastapi-service:8081/api/v1"
GERMANY_GEOJSON_URL = (
    "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/"
    "main/2_bundeslaender/4_niedrig.geo.json"
)

ENTGELDATLAS_BASELINES: Dict[str, Dict[str, int]] = {
    "Data Engineer (43214)": {"p25": 55_200, "median": 72_400, "p75": 88_900},
    "Data Scientist (43104)": {"p25": 58_100, "median": 75_600, "p75": 92_300},
    "Software Engineer (43414)": {"p25": 52_400, "median": 68_800, "p75": 85_100},
}

CITIES_COORDS: Dict[str, Dict[str, float]] = {
    "Berlin": {"lat": 52.5200, "lon": 13.4050},
    "München": {"lat": 48.1351, "lon": 11.5820},
    "Munich": {"lat": 48.1351, "lon": 11.5820},
    "Hamburg": {"lat": 53.5511, "lon": 9.9937},
    "Frankfurt am Main": {"lat": 50.1109, "lon": 8.6821},
    "Frankfurt": {"lat": 50.1109, "lon": 8.6821},
    "Köln": {"lat": 50.9375, "lon": 6.9603},
    "Cologne": {"lat": 50.9375, "lon": 6.9603},
    "Stuttgart": {"lat": 48.7758, "lon": 9.1829},
    "Düsseldorf": {"lat": 51.2277, "lon": 6.7735},
    "Dusseldorf": {"lat": 51.2277, "lon": 6.7735},
    "Dortmund": {"lat": 51.5136, "lon": 7.4653},
    "Essen": {"lat": 51.4556, "lon": 7.0116},
    "Leipzig": {"lat": 51.3397, "lon": 12.3731},
    "Bremen": {"lat": 53.0793, "lon": 8.8017},
    "Dresden": {"lat": 51.0504, "lon": 13.7373},
    "Hannover": {"lat": 52.3759, "lon": 9.7320},
    "Nürnberg": {"lat": 49.4521, "lon": 11.0767},
    "Nuremberg": {"lat": 49.4521, "lon": 11.0767},
    "Duisburg": {"lat": 51.4344, "lon": 6.7623},
    "Bochum": {"lat": 51.4818, "lon": 7.2162},
    "Wuppertal": {"lat": 51.2562, "lon": 7.1508},
    "Bonn": {"lat": 50.7374, "lon": 7.0982},
    "Bielefeld": {"lat": 52.0302, "lon": 8.5325},
    "Mannheim": {"lat": 49.4875, "lon": 8.4660},
    "Karlsruhe": {"lat": 49.0069, "lon": 8.4037},
    "Wiesbaden": {"lat": 50.0782, "lon": 8.2397},
    "Münster": {"lat": 51.9607, "lon": 7.6261},
    "Aachen": {"lat": 50.7753, "lon": 6.0839},
    "Freiburg": {"lat": 47.9990, "lon": 7.8421},
    "Ulm": {"lat": 48.4011, "lon": 9.9876},
    "Bruchsal": {"lat": 49.1259, "lon": 8.5997},
}

GERMAN_STATES: List[str] = [
    "Baden-Württemberg",
    "Bayern",
    "Berlin",
    "Brandenburg",
    "Bremen",
    "Hamburg",
    "Hessen",
    "Mecklenburg-Vorpommern",
    "Niedersachsen",
    "Nordrhein-Westfalen",
    "Rheinland-Pfalz",
    "Saarland",
    "Sachsen",
    "Sachsen-Anhalt",
    "Schleswig-Holstein",
    "Thüringen",
]

# Maps cities (and common English/state aliases) to official GeoJSON Bundesland names
CITY_TO_STATE: Dict[str, str] = {
    "Berlin": "Berlin",
    "München": "Bayern",
    "Munich": "Bayern",
    "Nürnberg": "Bayern",
    "Nuremberg": "Bayern",
    "Augsburg": "Bayern",
    "Regensburg": "Bayern",
    "Erlangen": "Bayern",
    "Würzburg": "Bayern",
    "Bavaria": "Bayern",
    "Hamburg": "Hamburg",
    "Frankfurt am Main": "Hessen",
    "Frankfurt": "Hessen",
    "Wiesbaden": "Hessen",
    "Darmstadt": "Hessen",
    "Kassel": "Hessen",
    "Hesse": "Hessen",
    "Köln": "Nordrhein-Westfalen",
    "Cologne": "Nordrhein-Westfalen",
    "Düsseldorf": "Nordrhein-Westfalen",
    "Dusseldorf": "Nordrhein-Westfalen",
    "Dortmund": "Nordrhein-Westfalen",
    "Essen": "Nordrhein-Westfalen",
    "Duisburg": "Nordrhein-Westfalen",
    "Bochum": "Nordrhein-Westfalen",
    "Wuppertal": "Nordrhein-Westfalen",
    "Bonn": "Nordrhein-Westfalen",
    "Bielefeld": "Nordrhein-Westfalen",
    "Münster": "Nordrhein-Westfalen",
    "Aachen": "Nordrhein-Westfalen",
    "Stuttgart": "Baden-Württemberg",
    "Mannheim": "Baden-Württemberg",
    "Karlsruhe": "Baden-Württemberg",
    "Freiburg": "Baden-Württemberg",
    "Ulm": "Baden-Württemberg",
    "Bruchsal": "Baden-Württemberg",
    "Heidelberg": "Baden-Württemberg",
    "Leipzig": "Sachsen",
    "Dresden": "Sachsen",
    "Chemnitz": "Sachsen",
    "Bremen": "Bremen",
    "Hannover": "Niedersachsen",
    "Hanover": "Niedersachsen",
    "Braunschweig": "Niedersachsen",
    "Wolfsburg": "Niedersachsen",
    "Potsdam": "Brandenburg",
    "Kiel": "Schleswig-Holstein",
    "Lübeck": "Schleswig-Holstein",
    "Rostock": "Mecklenburg-Vorpommern",
    "Schwerin": "Mecklenburg-Vorpommern",
    "Mainz": "Rheinland-Pfalz",
    "Koblenz": "Rheinland-Pfalz",
    "Saarbrücken": "Saarland",
    "Magdeburg": "Sachsen-Anhalt",
    "Halle": "Sachsen-Anhalt",
    "Erfurt": "Thüringen",
    "Jena": "Thüringen",
    **{state: state for state in GERMAN_STATES},
}


# -----------------------------------------------------------------------------
# DATA LOADING & HELPER FUNCTIONS
# -----------------------------------------------------------------------------


@st.cache_data
def load_job_titles() -> List[str]:
    file_path = "/app/tech_jobs_dictionary.json"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return list(data.keys()) if isinstance(data, dict) else data
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not parse JSON: {exc}")
    return list(ENTGELDATLAS_BASELINES.keys())


def safe_parse(val) -> List[str]:
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    if isinstance(val, str):
        v = val.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                parsed = ast.literal_eval(v)
                return parsed if isinstance(parsed, list) else []
            except (ValueError, SyntaxError):
                pass
        return [
            s.strip().strip("'").strip('"')
            for s in v.strip("[]").split(",")
            if s.strip()
        ]
    return []


def _get_auth_headers() -> Dict[str, str]:
    service_token = os.getenv("API_SECRET_KEY")
    if not service_token:
        return {}
    return {
        "accept": "application/json",
        "Authorization": f"Bearer {service_token}",
    }


@st.cache_data(ttl=600)
def fetch_all_jobs_from_fastapi() -> pd.DataFrame:
    """Fetches ALL clustered job data securely via FastAPI Endpoint to calculate global metrics."""
    headers = _get_auth_headers()
    if not headers:
        st.error("Missing API_SECRET_KEY environment variable.")
        return pd.DataFrame()

    try:
        response = requests.get(
            f"{FASTAPI_BASE_URL}/jobs?limit=10000",
            headers=headers,
            timeout=15,
        )
        if response.status_code != 200:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return pd.DataFrame()

        payload = response.json()
        records = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if "tech_stack" not in df.columns:
            df["tech_stack"] = [[] for _ in range(len(df))]
        df["tech_stack"] = df["tech_stack"].apply(safe_parse)

        if "location" in df.columns and "location_city" not in df.columns:
            df["location_city"] = df["location"]

        return df
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to reach FastAPI service: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_metrics_from_fastapi() -> pd.DataFrame:
    headers = _get_auth_headers()
    if not headers:
        return pd.DataFrame()
    try:
        response = requests.get(
            f"{FASTAPI_BASE_URL}/ml_metrics",
            headers=headers,
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data) if data else pd.DataFrame()
    except requests.exceptions.RequestException:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_salary_data_from_fastapi() -> dict:
    headers = _get_auth_headers()
    if not headers:
        return {"salaryData": ENTGELDATLAS_BASELINES, "source": "fallback"}
    try:
        response = requests.get(
            f"{FASTAPI_BASE_URL}/salary",
            headers=headers,
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            if isinstance(data, dict) and "salaryData" in data:
                return {"salaryData": data["salaryData"], "source": "api"}
            return {"salaryData": data, "source": "api"}
    except requests.exceptions.RequestException:
        pass
    return {"salaryData": ENTGELDATLAS_BASELINES, "source": "fallback"}


@st.cache_data(ttl=86400)
def load_germany_geojson() -> dict:
    """Fetches and caches Germany's 16 Bundesländer GeoJSON boundaries."""
    try:
        response = requests.get(GERMANY_GEOJSON_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        st.warning(f"Could not load German state boundaries: {exc}")
        return {}


def compute_centroids(df: pd.DataFrame) -> pd.DataFrame:
    if "tech_stack" not in df.columns or "cluster_id" not in df.columns:
        return pd.DataFrame()
    valid_df = df[df["tech_stack"].apply(len) > 0]
    if valid_df.empty:
        return pd.DataFrame()
    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(valid_df["tech_stack"])
    feature_df = pd.DataFrame(
        binary_matrix,
        columns=[str(col) for col in mlb.classes_],
        index=valid_df.index,
    )
    feature_df["cluster_id"] = df.loc[feature_df.index, "cluster_id"]
    return feature_df.groupby("cluster_id").mean()


def get_role_regex(role_name: str) -> str:
    r = role_name.lower()
    if "software" in r:
        return "software|developer|programmer|backend|frontend|fullstack|entwickler|swe"
    if "data engineer" in r:
        return "data engineer|data integration|big data|data platform|data architecture"
    if "data scientist" in r:
        return "data scientist|data science|machine learning|ml|ai|artificial intelligence"
    if "data analyst" in r:
        return "data analyst|analytics|business analyst|bi |business intelligence"
    if "cloud" in r:
        return "cloud|aws|azure|gcp"
    if "machine learning" in r:
        return "machine learning|ml|ai|artificial intelligence|computer vision|nlp"
    if "devops" in r:
        return "devops|sre|site reliability|platform|infrastructure"
    if "backend" in r:
        return "backend|back-end|back end|java|c#|golang|python"
    return role_name.split(" ")[0]


def normalize_is_remote(val) -> str:
    if pd.isna(val):
        return "Unspecified"
    if isinstance(val, bool):
        return "Remote" if val else "On-site"
    if isinstance(val, (int, float)):
        return "Remote" if val == 1 else "On-site"
    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "remote"}:
        return "Remote"
    if s in {"false", "0", "no", "on-site", "onsite"}:
        return "On-site"
    return "Unspecified"


# -----------------------------------------------------------------------------
# STREAMLIT APPLICATION INTERFACE
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="German Tech Role Market Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Configuration")

# 1. Fetch ALL data first to generate counts
all_jobs_master = fetch_all_jobs_from_fastapi()

if all_jobs_master.empty:
    st.warning(
        "⚠️ No data found in the database. Ensure the Airflow pipeline has run "
        "and FastAPI is accessible."
    )
    st.stop()

# Calculate jobs per role using robust regex mapping
job_options = load_job_titles()
role_counts: Dict[str, int] = {}
for role in job_options:
    regex_pattern = get_role_regex(role)
    if "job_title" in all_jobs_master.columns:
        count = int(
            all_jobs_master["job_title"]
            .str.contains(regex_pattern, case=False, na=False, regex=True)
            .sum()
        )
    else:
        count = 0
    role_counts[role] = count

formatted_options = [f"{role} ({role_counts[role]} jobs)" for role in job_options]
selected_formatted_role = st.sidebar.radio("Occupational Target", options=formatted_options)
selected_role = selected_formatted_role.split(" (")[0]

st.sidebar.markdown("---")
st.sidebar.subheader("Job Postings Filters")

if "job_title" in all_jobs_master.columns:
    selected_regex = get_role_regex(selected_role)
    all_data = all_jobs_master[
        all_jobs_master["job_title"].str.contains(
            selected_regex, case=False, na=False, regex=True
        )
    ].copy()
else:
    all_data = pd.DataFrame()

# 2. Resolve seniority column consistently
sen_col = next((c for c in ["seniority_level", "level"] if c in all_data.columns), None)

st.sidebar.markdown("**Seniority Level**")
if sen_col:
    available_seniorities = sorted(
        all_data[sen_col].fillna("Unspecified").astype(str).unique()
    )
    selected_seniority = [
        sen
        for sen in available_seniorities
        if st.sidebar.checkbox(f"Seniority: {sen}", value=True, key=f"sen_{sen}")
    ]
else:
    st.sidebar.caption("No seniority data available.")
    selected_seniority = None

st.sidebar.markdown("---")

# 3. Resolve work model (derive from is_remote if work_model is missing)
st.sidebar.markdown("**Work Model**")
if "work_model" not in all_data.columns and "is_remote" in all_data.columns:
    all_data["work_model"] = all_data["is_remote"].apply(normalize_is_remote)

if "work_model" in all_data.columns:
    available_models = sorted(
        all_data["work_model"].fillna("Unspecified").astype(str).unique()
    )
    selected_work_models = [
        model
        for model in available_models
        if st.sidebar.checkbox(f"Model: {model}", value=True, key=f"mod_{model}")
    ]
else:
    st.sidebar.caption("No work model data available.")
    selected_work_models = None

# 4. Apply filters (even when a selected list is empty)
filtered_jobs = all_data.copy()
if sen_col and selected_seniority is not None:
    filtered_jobs = filtered_jobs[
        filtered_jobs[sen_col].fillna("Unspecified").astype(str).isin(selected_seniority)
    ]
if "work_model" in filtered_jobs.columns and selected_work_models is not None:
    filtered_jobs = filtered_jobs[
        filtered_jobs["work_model"]
        .fillna("Unspecified")
        .astype(str)
        .isin(selected_work_models)
    ]

# MAIN UI - GENERAL PURPOSE CLEAN LAYOUT
st.title("German Tech Role Market Segmentation")

api_response = fetch_salary_data_from_fastapi()
salary_data = api_response.get("salaryData", {})

salary_stats = {}
matched_baseline_key = None
for key in ENTGELDATLAS_BASELINES:
    if selected_role.lower() in key.lower() or key.lower() in selected_role.lower():
        matched_baseline_key = key
        if key in salary_data:
            salary_stats = salary_data[key]
        else:
            code_match = re.search(r"\((.*?)\)", key)
            if code_match and code_match.group(1) in salary_data:
                salary_stats = salary_data[code_match.group(1)]
        break

# Clean Top Metrics Bar (MacBook Optimized, Compact)
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Total Database", f"{len(all_jobs_master):,}")
col_m2.metric(f"Role: {selected_role}", f"{role_counts.get(selected_role, 0):,}")
col_m3.metric("Matching Filters", f"{len(filtered_jobs):,}")

if matched_baseline_key:
    fallback_median = ENTGELDATLAS_BASELINES.get(matched_baseline_key, {}).get("median", 68_800)
    median_sal = f"€{salary_stats.get('median', fallback_median):,}"
else:
    median_sal = "N/A"
col_m4.metric("National Median Salary", median_sal)

if api_response.get("source") == "api":
    st.caption("✅ Salary baselines loaded from live API.")
else:
    st.caption("⚠️ Salary baselines loaded from offline fallback.")

st.markdown("---")

if len(filtered_jobs) < 3 or "cluster_id" not in filtered_jobs.columns:
    st.warning(
        "Not enough job postings match your filters to render clustering visuals. "
        "Please broaden your search."
    )
else:
    centroid_matrix = compute_centroids(filtered_jobs)

    dynamic_labels = {}
    if not centroid_matrix.empty:
        for idx, row in centroid_matrix.iterrows():
            top_skills = row.nlargest(3).index.tolist()
            if top_skills:
                dynamic_labels[idx] = f"Cluster {idx}: " + " / ".join(top_skills)
            else:
                dynamic_labels[idx] = f"Cluster {idx}: General Skills"

        centroid_matrix.index = centroid_matrix.index.map(dynamic_labels)

    if "cluster_name" not in filtered_jobs.columns:
        filtered_jobs["cluster_name"] = (
            filtered_jobs["cluster_id"].map(dynamic_labels).fillna("General Cluster")
        )

    tab_heatmap, tab_deepdive, tab_geo, tab_monitoring = st.tabs(
        [
            "📊 Skill Heatmap & Distribution",
            "🔍 Archetype Deep Dive",
            "🗺️ Geographic Hubs",
            "📈 ML Monitoring",
        ]
    )

    with tab_heatmap:
        st.subheader("Market Archetypes & Skill Prevalence")
        sub_chart_type, sub_volumes = st.tabs(
            ["Skill Heatmap", "Cluster Volume Distribution"]
        )

        with sub_chart_type:
            if not centroid_matrix.empty:
                top_skills = centroid_matrix.mean().nlargest(
                    min(12, centroid_matrix.shape[1])
                ).index
                plot_matrix = centroid_matrix[top_skills]
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.heatmap(
                    plot_matrix,
                    cmap="YlGnBu",
                    annot=True,
                    fmt=".2f",
                    cbar_kws={"label": "Probability"},
                    ax=ax,
                    annot_kws={"size": 8},
                    linewidths=0.5,
                )
                ax.set_ylabel("Cluster Archetype", fontsize=9)
                ax.set_xlabel("Extracted Skills", fontsize=9)
                plt.xticks(rotation=40, ha="right", fontsize=8)
                plt.yticks(fontsize=8)
                plt.subplots_adjust(bottom=0.32, left=0.28, right=0.95, top=0.92)
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Insufficient skill metrics for heatmap generation.")

        with sub_volumes:
            vol_series = filtered_jobs["cluster_name"].value_counts().sort_values(ascending=True)
            fig_vol, ax_vol = plt.subplots(figsize=(10, 4))
            vol_series.plot(kind="barh", ax=ax_vol, color="#2b5c8f")
            ax_vol.set_xlabel("Number of Job Postings", fontsize=9)
            ax_vol.set_ylabel("Cluster Archetype", fontsize=9)
            ax_vol.tick_params(axis="both", labelsize=8)
            plt.subplots_adjust(bottom=0.20, left=0.35, right=0.95, top=0.92)
            st.pyplot(fig_vol, clear_figure=True)

    with tab_deepdive:
        st.subheader("Archetype Deep Dive & Raw Postings")
        active_archetypes = list(filtered_jobs["cluster_name"].dropna().unique())

        if active_archetypes:
            selected_archetype = st.radio(
                "Select Cluster to Inspect:",
                options=active_archetypes,
                horizontal=True,
            )
            cluster_slice = filtered_jobs[
                filtered_jobs["cluster_name"] == selected_archetype
            ].copy()

            st.markdown(f"**Top Hubs for {selected_archetype}**")
            loc_col = "location_city" if "location_city" in cluster_slice.columns else "location"
            if loc_col in cluster_slice.columns:
                cleaned_locations = (
                    cluster_slice[loc_col]
                    .astype(str)
                    .apply(lambda x: x.split(",")[0].strip())
                )
                geo_counts = cleaned_locations.value_counts().reset_index()
                geo_counts.columns = ["City", "Vacancies"]
                st.dataframe(geo_counts.head(10).set_index("City").T, use_container_width=True)
            else:
                st.write("Location data unavailable.")

            st.markdown("---")
            st.markdown(
                f"**All Raw Postings in {selected_archetype} ({len(cluster_slice)} jobs)**"
            )

            if "tech_stack" not in cluster_slice.columns:
                cluster_slice["tech_stack"] = [[] for _ in range(len(cluster_slice))]

            display_cols = [
                c
                for c in ["company_name", "job_title", "location_city", "work_model", "tech_stack"]
                if c in cluster_slice.columns
            ]

            st.dataframe(
                cluster_slice[display_cols],
                use_container_width=True,
                hide_index=True,
                height=600,
                column_config={
                    "tech_stack": st.column_config.ListColumn("Tech Stack", width="large"),
                    "job_title": st.column_config.TextColumn("Job Title", width="medium"),
                    "company_name": st.column_config.TextColumn("Company", width="medium"),
                },
            )

    with tab_geo:
        st.subheader(f"Interactive Geographic Map & Hubs: {selected_role}")
        st.caption(
            "ℹ️ **Interactive Map & Data Note:** Hover over or click on the 3D columns "
            "on the map to explore job density across German tech hubs. Listings "
            "specifying only country-level metadata are aggregated separately."
        )

        loc_col = "location_city" if "location_city" in filtered_jobs.columns else "location"
        if loc_col in filtered_jobs.columns and len(filtered_jobs) > 0:
            cleaned_locations = (
                filtered_jobs[loc_col]
                .astype(str)
                .apply(lambda x: x.split(",")[0].strip())
            )
            geo_counts = cleaned_locations.value_counts().reset_index()
            geo_counts.columns = ["City", "Total Vacancies"]

            map_data = [
                {
                    "city": row["City"],
                    "lat": CITIES_COORDS[row["City"]]["lat"],
                    "lon": CITIES_COORDS[row["City"]]["lon"],
                    "vacancies": row["Total Vacancies"],
                    "elevation": row["Total Vacancies"] * 150,
                }
                for _, row in geo_counts.iterrows()
                if row["City"] in CITIES_COORDS
            ]
            map_df = pd.DataFrame(map_data)

            if not map_df.empty:
                layer = pdk.Layer(
                    "ColumnLayer",
                    map_df,
                    get_position=["lon", "lat"],
                    get_elevation="elevation",
                    elevation_scale=10,
                    radius=25000,
                    get_fill_color=[43, 92, 143, 180],
                    pickable=True,
                    auto_highlight=True,
                )

                tooltip = {
                    "html": "<b>City:</b> {city} <br/><b>Job Vacancies:</b> {vacancies}",
                    "style": {"backgroundColor": "steelblue", "color": "white"},
                }

                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=pdk.ViewState(
                        latitude=51.1657,
                        longitude=10.4515,
                        zoom=5.5,
                        pitch=40.5,
                    ),
                    tooltip=tooltip,
                )
                st.pydeck_chart(deck)
            else:
                st.info("Mapping coordinates available for primary hubs.")

            # 2D Choropleth Map by Federal State (Bundesland)
            st.markdown("---")
            st.markdown("### 🗺️ 2D Regional Overview by Federal State (*Bundesland*)")
            st.caption(
                "Hover over any German state to see total vacancies and top contributing cities."
            )

            germany_geojson = load_germany_geojson()
            if germany_geojson:
                geo_counts["State"] = geo_counts["City"].map(CITY_TO_STATE)
                mapped = geo_counts.dropna(subset=["State"])

                state_totals = mapped.groupby("State")["Total Vacancies"].sum()
                top_cities_by_state = (
                    mapped.sort_values("Total Vacancies", ascending=False)
                    .groupby("State")
                    .apply(
                        lambda g: ", ".join(
                            f"{r['City']} ({r['Total Vacancies']})"
                            for _, r in g.head(3).iterrows()
                        )
                    )
                )

                state_df = pd.DataFrame({"State": GERMAN_STATES})
                state_df["Total Vacancies"] = (
                    state_df["State"].map(state_totals).fillna(0).astype(int)
                )
                state_df["Top Cities"] = (
                    state_df["State"].map(top_cities_by_state).fillna("No mapped listings")
                )

                fig_2d = px.choropleth(
                    state_df,
                    geojson=germany_geojson,
                    locations="State",
                    featureidkey="properties.name",
                    color="Total Vacancies",
                    color_continuous_scale="YlGnBu",
                    hover_name="State",
                    hover_data={
                        "State": False,
                        "Total Vacancies": True,
                        "Top Cities": True,
                    },
                )
                fig_2d.update_geos(
                    fitbounds="locations",
                    visible=False,
                    bgcolor="rgba(0,0,0,0)",
                )
                fig_2d.update_layout(
                    margin={"r": 0, "t": 10, "l": 0, "b": 0},
                    height=550,
                    paper_bgcolor="rgba(0,0,0,0)",
                    coloraxis_colorbar_title="Vacancies",
                )
                st.plotly_chart(fig_2d, use_container_width=True)

                unmapped_count = int(
                    geo_counts[geo_counts["State"].isna()]["Total Vacancies"].sum()
                )
                if unmapped_count > 0:
                    st.caption(
                        f"ℹ️ {unmapped_count:,} listing(s) had locations not mapped to a "
                        "federal state (e.g., country-level 'Germany', 'Remote', or smaller "
                        "towns) and are excluded from the state totals."
                    )

            st.markdown("---")
            st.dataframe(geo_counts, use_container_width=True, height=400, hide_index=True)
        else:
            st.info("No location data available.")

    with tab_monitoring:
        st.subheader("ML Model Performance & Drift Tracking")

        st.markdown(
            """
            ### 📖 Understanding Clustering Metrics
            * **Inertia (Within-Cluster Sum of Squares):** Measures how tightly clustered data points are. Lower values indicate denser, more cohesive clusters.
            * **Silhouette Score:** Measures cluster separation ranging from $-1$ to $+1$ (values $> 0.25$ indicate reasonable separation, $> 0.50$ indicate strong structures).
            """
        )

        st.markdown("---")

        metrics_df = fetch_metrics_from_fastapi()
        if metrics_df.empty:
            st.info(
                "ℹ️ No historical metrics found. They will appear here once your "
                "Airflow clustering DAG executes its next run."
            )
        else:
            if "execution_date" in metrics_df.columns:
                metrics_df = metrics_df.copy()
                metrics_df["_parsed_date"] = pd.to_datetime(
                    metrics_df["execution_date"], format="ISO8601", utc=True, errors="coerce"
                )
                metrics_df = metrics_df.sort_values("_parsed_date", ascending=True).reset_index(drop=True)

            latest_row = metrics_df.iloc[-1]
            latest_version = latest_row.get("model_version", "unknown")
            sil_col = next(
                (c for c in ["silhouette_score", "silhouette"] if c in metrics_df.columns),
                None,
            )

            if sil_col and pd.notna(latest_row.get(sil_col)):
                sil_val = float(latest_row[sil_col])
                st.markdown(f"### 🚦 Current Model Health Evaluation (`{latest_version}`)")
                col_h1, col_h2 = st.columns(2)

                with col_h1:
                    if sil_val >= 0.35:
                        st.success(
                            f"**Silhouette Score: {sil_val:.3f}** — 🟢 **Excellent Quality**: "
                            "Clusters are well-defined and distinct."
                        )
                    elif sil_val >= 0.20:
                        st.warning(
                            f"**Silhouette Score: {sil_val:.3f}** — 🟡 **Moderate Quality**: "
                            "Acceptable overlap between skill clusters."
                        )
                    else:
                        st.error(
                            f"**Silhouette Score: {sil_val:.3f}** — 🔴 **Poor Quality**: "
                            "Clusters overlap significantly; consider adjusting features or $k$."
                        )

                with col_h2:
                    if "inertia" in latest_row and pd.notna(latest_row["inertia"]):
                        inertia_val = float(latest_row["inertia"])
                        st.info(
                            f"**Latest Inertia: {inertia_val:,.1f}** — 🔵 "
                            f"Recorded on model `{latest_version}`."
                        )

            st.markdown("---")
            st.markdown("### 📋 Recent Model Runs Log")
            display_metrics = metrics_df.drop(columns=["_parsed_date"], errors="ignore")
            st.dataframe(display_metrics, use_container_width=True, hide_index=True)

            if "_parsed_date" in metrics_df.columns:
                chart_df = metrics_df.dropna(subset=["_parsed_date"]).set_index("_parsed_date")
                if "inertia" in chart_df.columns:
                    st.markdown("### 📉 Inertia Trend Over Time")
                    st.line_chart(chart_df["inertia"])
                if sil_col:
                    st.markdown("### 📈 Silhouette Score Trend Over Time")
                    st.line_chart(chart_df[sil_col])
