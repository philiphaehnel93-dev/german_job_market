"""
Market Segmentation & National Baseline Dashboard
Analyzes German Tech Roles using Entgeldatlas Baselines and FastAPI Clustered Data.
"""

import ast
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pydeck as pdk
import requests
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st

st.set_page_config(
    page_title="German Tech Role Market Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import (
    CITIES_COORDS,
    CITY_TO_STATE,
    ENTGELDATLAS_BASELINES,
    FASTAPI_BASE_URL,
    GERMAN_STATES,
    GERMANY_GEOJSON_URL,
    JOB_ROLE_MAPPINGS,
    JOB_TITLES_DICT_PATH,
    SMALL_CLUSTER_THRESHOLD,
)


# -----------------------------------------------------------------------------
# DATA LOADING & HELPER FUNCTIONS
# -----------------------------------------------------------------------------


def _strip_role_code(label: str) -> str:
    """Removes a trailing numeric/parenthetical code such as ' (43214)' if present."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(label)).strip()


def _resolve_job_titles_path() -> str:
    """Finds tech_jobs_dictionary.json across Docker, local, and Streamlit Cloud layouts."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    candidates = [
        JOB_TITLES_DICT_PATH,
        os.path.join(current_dir, "tech_jobs_dictionary.json"),
        os.path.join(parent_dir, "tech_jobs_dictionary.json"),
        os.path.join(parent_dir, "dashboard", "tech_jobs_dictionary.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


@st.cache_data
def load_job_titles() -> List[str]:
    """
    Loads all jobs from tech_jobs_dictionary.json, grouping overlapping roles via
    JOB_ROLE_MAPPINGS from constants.py while keeping all unmapped jobs visible as unique roles.
    """
    raw_titles: List[str] = list(ENTGELDATLAS_BASELINES.keys())
    dict_path = _resolve_job_titles_path()
    if dict_path:
        try:
            with open(dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_titles = list(data.keys()) if isinstance(data, dict) else list(data)
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not parse JSON: {exc}")

    seen = set()
    unique_titles: List[str] = []
    for title in raw_titles:
        clean = str(title).strip()
        if not clean:
            continue
        # Group overlapping roles defined in JOB_ROLE_MAPPINGS; keep all others as-is
        canonical = (
            JOB_ROLE_MAPPINGS.get(clean.lower())
            or JOB_ROLE_MAPPINGS.get(_strip_role_code(clean).lower())
            or clean
        )
        canonical = str(canonical).strip()
        key = canonical.lower()
        if canonical and key not in seen:
            seen.add(key)
            unique_titles.append(canonical)
    return unique_titles


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


def compute_centroids(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Computes cluster skill prevalence centroids, job-weighted global skill prevalence,
    and cluster sizes (ordered from largest to smallest cluster).
    """
    if "tech_stack" not in df.columns or "cluster_id" not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int)

    valid_df = df[df["tech_stack"].apply(len) > 0]
    if valid_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int)

    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(valid_df["tech_stack"])
    feature_df = pd.DataFrame(
        binary_matrix,
        columns=[str(col) for col in mlb.classes_],
        index=valid_df.index,
    )

    global_means = feature_df.mean(axis=0)
    feature_df["cluster_id"] = valid_df["cluster_id"]

    cluster_sizes = df["cluster_id"].value_counts()
    centroids = feature_df.groupby("cluster_id").mean()

    ordered_ids = [cid for cid in cluster_sizes.index if cid in centroids.index]
    centroids = centroids.loc[ordered_ids]
    cluster_sizes = cluster_sizes.loc[ordered_ids]

    return centroids, global_means, cluster_sizes


def build_distinctive_labels(
    centroids: pd.DataFrame,
    global_means: pd.Series,
    cluster_sizes: pd.Series,
    top_n: int = 3,
    min_prevalence: float = 0.15,
) -> Dict[int, str]:
    """
    Builds interpretable cluster labels using distinctive lift over the role average
    plus exact sample size (n=...), avoiding repetitive generic labels.
    """
    labels: Dict[int, str] = {}
    if centroids.empty:
        return labels

    lift_matrix = centroids.sub(global_means, axis=1)

    for cid, row in centroids.iterrows():
        n_jobs = int(cluster_sizes.get(cid, 0))
        eligible = row[row >= min_prevalence].index
        if len(eligible) >= top_n:
            distinctive_skills = lift_matrix.loc[cid, eligible].nlargest(top_n).index.tolist()
        else:
            distinctive_skills = row[row > 0].nlargest(top_n).index.tolist()

        skill_str = " / ".join(distinctive_skills) if distinctive_skills else "General Skills"
        labels[cid] = f"Cluster {cid} (n={n_jobs:,}): {skill_str}"

    return labels


def get_role_regex(role_name: str) -> str:
    """
    Maps occupational targets to accurate title regex patterns.
    Includes any aliases mapped to `role_name` in JOB_ROLE_MAPPINGS while keeping
    unmapped roles matched by their own distinct title.
    """
    base = _strip_role_code(role_name).lower().strip()
    patterns: List[str] = []

    if "business intelligence" in base or base.startswith("bi "):
        patterns.append(r"business intelligence|\bbi\b|power bi|tableau|qlik")
    elif "analytics engineer" in base:
        patterns.append(r"analytics engineer|dbt|data modeling")
    elif "data architect" in base:
        patterns.append(r"data architect|datenarchitekt|enterprise architect")
    elif "data engineer" in base:
        patterns.append(r"data engineer|data integration|big data|data platform|etl")
    elif "data scientist" in base:
        patterns.append(r"data scientist|data science")
    elif "machine learning" in base or "ml engineer" in base:
        patterns.append(r"machine learning|\bml\b|mlops|ai engineer|computer vision|\bnlp\b")
    elif "data analyst" in base:
        patterns.append(r"data analyst|datenanalyst|business analyst")
    elif "cloud" in base:
        patterns.append(r"cloud engineer|cloud architect|\baws\b|\bazure\b|\bgcp\b")
    elif "devops" in base:
        patterns.append(r"devops|\bsre\b|site reliability|platform engineer|infrastructure engineer")
    elif "backend" in base:
        patterns.append(r"backend|back-end|back end")
    elif "software" in base:
        patterns.append(r"software|developer|programmer|frontend|fullstack|entwickler|\bswe\b")
    elif "deploy forward" in base or "forward deployed" in base:
        patterns.append(r"deploy forward|forward deployed")
    else:
        patterns.append(re.escape(base))

    # Also include any aliases explicitly grouped into this role via JOB_ROLE_MAPPINGS
    mapped_aliases = [
        re.escape(alias.strip().lower())
        for alias, target in JOB_ROLE_MAPPINGS.items()
        if str(target).strip().lower() in {role_name.strip().lower(), base}
        and alias.strip()
    ]
    for alias_pat in mapped_aliases:
        if alias_pat not in patterns:
            patterns.append(alias_pat)

    return "|".join(patterns)


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

# Balanced sidebar styling + safe top padding (prevents KPI clipping & removes extra gaps)
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.6rem;
            padding-bottom: 1.5rem;
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 0.9rem;
        }
        section[data-testid="stSidebar"] h3 {
            font-size: 1.15rem;
            margin-bottom: 0.35rem;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stRadio p,
        section[data-testid="stSidebar"] .stCheckbox p {
            font-size: 0.90rem !important;
            line-height: 1.3 !important;
        }
        section[data-testid="stSidebar"] .stRadio > div {
            gap: 0.22rem !important;
        }
        section[data-testid="stSidebar"] .stCheckbox {
            margin-bottom: -0.45rem !important;
        }
        section[data-testid="stSidebar"] hr {
            margin: 0.65rem 0 !important;
        }
        .dashboard-header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.55rem;
        }
        .dashboard-title {
            font-size: 1.45rem;
            font-weight: 700;
            color: #1f2937;
            margin: 0;
            line-height: 1.2;
        }
        .dashboard-badge {
            font-size: 0.8rem;
            color: #6b7280;
        }
        div[data-testid="stMetric"] {
            padding: 0.2rem 0;
        }
        hr.compact-divider {
            margin: 0.45rem 0 0.75rem 0;
            border: none;
            border-top: 1px solid #e5e7eb;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Configuration")

# 1. Fetch ALL data first to generate counts
all_jobs_master = fetch_all_jobs_from_fastapi()

if all_jobs_master.empty:
    st.warning(
        "⚠️ No data found in the database. Ensure the Airflow pipeline has run "
        "and FastAPI is accessible."
    )
    st.stop()

# Keep all unique jobs from tech_jobs_dictionary.json (grouped only by JOB_ROLE_MAPPINGS)
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

formatted_options = [f"{role} ({role_counts[role]:,})" for role in job_options]
selected_formatted_role = st.sidebar.radio("Occupational Target", options=formatted_options)
selected_role = selected_formatted_role.rsplit(" (", 1)[0]

st.sidebar.markdown("---")

if "job_title" in all_jobs_master.columns:
    selected_regex = get_role_regex(selected_role)
    all_data = all_jobs_master[
        all_jobs_master["job_title"].str.contains(
            selected_regex, case=False, na=False, regex=True
        )
    ].copy()
else:
    all_data = pd.DataFrame()

# 2. Resolve seniority column consistently (click-only checkboxes)
sen_col = next((c for c in ["seniority_level", "level"] if c in all_data.columns), None)

st.sidebar.markdown("**Seniority Level**")
if sen_col:
    available_seniorities = sorted(
        all_data[sen_col].fillna("Unspecified").astype(str).unique()
    )
    selected_seniority = [
        sen
        for sen in available_seniorities
        if st.sidebar.checkbox(sen, value=True, key=f"sen_{sen}")
    ]
else:
    st.sidebar.caption("No seniority data available.")
    selected_seniority = None

st.sidebar.markdown("---")

# 3. Resolve work model (click-only checkboxes)
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
        if st.sidebar.checkbox(model, value=True, key=f"mod_{model}")
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

# -----------------------------------------------------------------------------
# COMPACT HEADER & TOP KPI / SALARY BAR
# -----------------------------------------------------------------------------

api_response = fetch_salary_data_from_fastapi()
salary_data = api_response.get("salaryData", {})

salary_stats = {}
matched_baseline_key = None
selected_role_clean = _strip_role_code(selected_role).lower()
for key in ENTGELDATLAS_BASELINES:
    key_lower = key.lower()
    if (
        selected_role.lower() in key_lower
        or key_lower in selected_role.lower()
        or (selected_role_clean and selected_role_clean in key_lower)
    ):
        matched_baseline_key = key
        if key in salary_data:
            salary_stats = salary_data[key]
        else:
            code_match = re.search(r"\((.*?)\)", key)
            if code_match and code_match.group(1) in salary_data:
                salary_stats = salary_data[code_match.group(1)]
        break

source_caption = (
    "✅ Salary baselines loaded from live API"
    if api_response.get("source") == "api"
    else "⚠️ Salary baselines loaded from offline fallback"
)

st.markdown(
    f"""
    <div class="dashboard-header">
        <h1 class="dashboard-title">German Tech Role Market Segmentation</h1>
        <span class="dashboard-badge">{source_caption}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Total Database", f"{len(all_jobs_master):,}")
col_m2.metric(f"Role: {selected_role}", f"{role_counts.get(selected_role, len(all_data)):,}")
col_m3.metric("Matching Filters", f"{len(filtered_jobs):,}")

if matched_baseline_key:
    fallback_median = ENTGELDATLAS_BASELINES.get(matched_baseline_key, {}).get("median", 68_800)
    median_sal = f"€{salary_stats.get('median', fallback_median):,}"
else:
    median_sal = "N/A"
col_m4.metric("National Median Salary", median_sal)

st.markdown('<hr class="compact-divider">', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN CONTENT TABS (STAKEHOLDER-FRIENDLY LABELS)
# -----------------------------------------------------------------------------

if len(filtered_jobs) < 3 or "cluster_id" not in filtered_jobs.columns:
    st.warning(
        "Not enough job postings match your filters to render clustering visuals. "
        "Please broaden your search."
    )
else:
    centroid_matrix, global_skill_means, cluster_sizes = compute_centroids(filtered_jobs)
    dynamic_labels = build_distinctive_labels(
        centroid_matrix, global_skill_means, cluster_sizes
    )

    if not centroid_matrix.empty:
        centroid_matrix.index = centroid_matrix.index.map(dynamic_labels)

    filtered_jobs["cluster_name"] = (
        filtered_jobs["cluster_id"].map(dynamic_labels).fillna("General Cluster")
    )

    tab_heatmap, tab_deepdive, tab_geo, tab_monitoring = st.tabs(
        [
            "📊 Skill Demand Overview",
            "🔍 Role Profiles & Listings",
            "🗺️ Hiring Locations",
            "📈 Model Health & Quality",
        ]
    )

    with tab_heatmap:
        st.subheader("Market Segments & Skill Demand")
        sub_chart_type, sub_volumes = st.tabs(
            ["Skill Demand Matrix", "Market Share by Segment"]
        )

        with sub_chart_type:
            if not centroid_matrix.empty and not global_skill_means.empty:
                top_skills = global_skill_means.nlargest(
                    min(12, len(global_skill_means))
                ).index
                plot_matrix = centroid_matrix[top_skills]

                fig, ax = plt.subplots(figsize=(10.5, max(3.6, 0.75 * len(plot_matrix) + 1.2)))
                sns.heatmap(
                    plot_matrix,
                    cmap="YlGnBu",
                    vmin=0.0,
                    vmax=1.0,
                    annot=True,
                    fmt=".2f",
                    cbar_kws={"label": "Share of Jobs in Segment"},
                    ax=ax,
                    annot_kws={"size": 8},
                    linewidths=0.5,
                )
                ax.set_ylabel("Market Segment (Distinctive Skills)", fontsize=9)
                ax.set_xlabel("Top Extracted Skills (Job-Weighted)", fontsize=9)
                plt.xticks(rotation=40, ha="right", fontsize=8)
                plt.yticks(fontsize=8)
                plt.subplots_adjust(bottom=0.30, left=0.34, right=0.95, top=0.92)
                st.pyplot(fig, clear_figure=True)

                small_clusters = cluster_sizes[cluster_sizes < SMALL_CLUSTER_THRESHOLD]
                if not small_clusters.empty:
                    st.caption(
                        f"ℹ️ Rows are ordered by segment volume. Segments with fewer than "
                        f"{SMALL_CLUSTER_THRESHOLD} postings represent niche/outlier profiles—interpret "
                        f"high percentages on small sample sizes cautiously."
                    )
            else:
                st.info("Insufficient skill metrics for heatmap generation.")

        with sub_volumes:
            vol_series = (
                filtered_jobs["cluster_name"]
                .value_counts()
                .sort_values(ascending=True)
            )
            total_vol = int(vol_series.sum())
            fig_vol, ax_vol = plt.subplots(
                figsize=(10.5, max(3.5, 0.7 * len(vol_series) + 1.0))
            )
            bars = vol_series.plot(kind="barh", ax=ax_vol, color="#2b5c8f")
            ax_vol.set_xlabel("Number of Job Postings", fontsize=9)
            ax_vol.set_ylabel("Market Segment", fontsize=9)
            ax_vol.tick_params(axis="both", labelsize=8)

            max_val = vol_series.max() if not vol_series.empty else 1
            ax_vol.set_xlim(0, max_val * 1.18)
            for patch, count in zip(bars.patches, vol_series.values):
                pct = (count / total_vol * 100) if total_vol > 0 else 0.0
                ax_vol.text(
                    patch.get_width() + max_val * 0.015,
                    patch.get_y() + patch.get_height() / 2,
                    f"{int(count):,} ({pct:.1f}%)",
                    va="center",
                    fontsize=8,
                    color="#1f2937",
                )

            plt.subplots_adjust(bottom=0.18, left=0.38, right=0.96, top=0.92)
            st.pyplot(fig_vol, clear_figure=True)

    with tab_deepdive:
        st.subheader("Role Profiles & Raw Job Listings")
        active_archetypes = (
            filtered_jobs["cluster_name"].value_counts().index.tolist()
        )

        if active_archetypes:
            selected_archetype = st.radio(
                "Select Segment to Inspect:",
                options=active_archetypes,
                horizontal=True,
            )
            cluster_slice = filtered_jobs[
                filtered_jobs["cluster_name"] == selected_archetype
            ].copy()

            st.markdown(f"**Top Hiring Cities for {selected_archetype}**")
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
                f"**All Job Listings in {selected_archetype} ({len(cluster_slice):,} jobs)**"
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
        st.subheader(f"Hiring Locations & Regional Demand: {selected_role}")
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
        st.subheader("Model Health & Quality Tracking")

        st.markdown(
            """
            ### 📖 Understanding Segmentation Quality Metrics
            * **Inertia (Within-Cluster Sum of Squares):** Measures how tightly grouped job postings are within each segment. Lower values indicate denser, more cohesive segments.
            * **Silhouette Score:** Measures how cleanly segments are separated from $-1$ to $+1$ (values $> 0.25$ indicate reasonable separation, $> 0.50$ indicate strong structure).
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
                            "Segments are well-defined and distinct."
                        )
                    elif sil_val >= 0.20:
                        st.warning(
                            f"**Silhouette Score: {sil_val:.3f}** — 🟡 **Moderate Quality**: "
                            "Acceptable overlap between skill segments."
                        )
                    else:
                        st.error(
                            f"**Silhouette Score: {sil_val:.3f}** — 🔴 **Poor Quality**: "
                            "Segments overlap significantly; consider adjusting features or $k$."
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
