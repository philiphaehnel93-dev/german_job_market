"""
Market Segmentation & National Baseline Dashboard
Analyzes German Tech Roles using Entgeldatlas Baselines and FastAPI Clustered Data.
"""

from typing import Dict, List
import os
import json
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st
import re

# -----------------------------------------------------------------------------
# DATA SOURCES & SCHEMAS
# -----------------------------------------------------------------------------

ENTGELDATLAS_BASELINES: Dict[str, Dict[str, int]] = {
    "Data Engineer (43214)": {"p25": 55_200, "median": 72_400, "p75": 88_900},
    "Data Scientist (43104)": {"p25": 58_100, "median": 75_600, "p75": 92_300},
    "Software Engineer (43414)": {"p25": 52_400, "median": 68_800, "p75": 85_100},
}

CLUSTER_LABELS: Dict[int, str] = {
    0: "Core Python & SQL Generalist",
    1: "Data Engineering & Analytics",
    2: "Cloud Data Engineering (Python/AWS/Azure)",
    3: "Lightweight Scripting & BI",
    4: "DevOps & Container Infrastructure",
    5: "Backend & Services",
    6: "Advanced ML & Architecture",
}

@st.cache_data
def load_job_titles() -> List[str]:
    """Loads job titles directly from the Airflow dictionary if available."""
    file_path = "/app/tech_jobs_dictionary.json"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return list(data.keys()) if isinstance(data, dict) else data
        except Exception as e:
            st.error(f"Could not parse JSON: {e}")

    return list(ENTGELDATLAS_BASELINES.keys())

@st.cache_data(ttl=600)
def fetch_job_data_from_fastapi() -> pd.DataFrame:
    """Fetches clustered job data securely via FastAPI Endpoint."""
    SERVICE_TOKEN = os.getenv("API_SECRET_KEY")

    if not SERVICE_TOKEN:
        st.error("Missing API_SECRET_KEY environment variable.")
        return pd.DataFrame()

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {SERVICE_TOKEN}"
        }

    try:
        response = requests.get(
            "http://fastapi-service:8081/api/v1/jobs",
            headers=headers,
            timeout=10,
        )

        if response.status_code != 200:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return pd.DataFrame()

        payload = response.json()
        records = payload.get("data", payload) if isinstance(payload, dict) else payload

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if "tech_stack" in df.columns:
            df = df[df["tech_stack"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        if "location" in df.columns and "location_city" not in df.columns:
            df["location_city"] = df["location"]

        return df
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to reach FastAPI service: {exc}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_metrics_from_fastapi() -> pd.DataFrame:
    """Fetches ML performance history securely via FastAPI."""
    SERVICE_TOKEN = os.getenv("API_SECRET_KEY")

    if not SERVICE_TOKEN:
        return pd.DataFrame()

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {SERVICE_TOKEN}"
    }

    try:
        response = requests.get(
            "http://fastapi-service:8081/api/v1/ml_metrics",
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
def fetch_salary_data_from_fastapi() -> pd.DataFrame:
    """Fetches salary data securely via FastAPI."""
    SERVICE_TOKEN = os.getenv("API_SECRET_KEY")

    default_salary = {
        "Data Engineer (43214)": {"p25": 55_200, "median": 72_400, "p75": 88_900},
        "Data Scientist (43104)": {"p25": 58_100, "median": 75_600, "p75": 92_300},
        "Software Engineer (43414)": {"p25": 52_400, "median": 68_800, "p75": 85_100}
    }

    if not SERVICE_TOKEN:
        return default_salary

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {SERVICE_TOKEN}"
    }

    try:
        response = requests.get(
            "http://fastapi-service:8081/api/v1/salary",
            headers=headers,
            params={"limit": 1},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            return data if data else default_salary
    except requests.exceptions.RequestException:
        pass

    return default_salary

def compute_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Computes skill probability centroids per cluster based on live data."""
    if "tech_stack" not in df.columns or "cluster_id" not in df.columns:
        return pd.DataFrame()

    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(df["tech_stack"].dropna())
    feature_df = pd.DataFrame(binary_matrix, columns=mlb.classes_, index=df.dropna(subset=["tech_stack"]).index)
    feature_df.columns = [str(col) for col in feature_df.columns]

    feature_df["cluster_id"] = df.loc[feature_df.index, "cluster_id"]
    centroids = feature_df.groupby("cluster_id").mean()
    centroids.index = [f"C{idx}: {CLUSTER_LABELS.get(idx, 'Archetype')}" for idx in centroids.index]
    return centroids

# -----------------------------------------------------------------------------
# STREAMLIT APPLICATION INTERFACE
# -----------------------------------------------------------------------------

st.set_page_config(page_title="German Tech Role Market Segmentation", layout="wide")

st.sidebar.title("Configuration")

job_options = load_job_titles()
selected_role = st.sidebar.radio("Occupational Target", options=job_options)

st.sidebar.markdown("---")
st.sidebar.subheader("Job Postings Filters")

# Automatically fetch data directly from FastAPI (button removed)
all_data = fetch_job_data_from_fastapi()

if all_data.empty:
    st.warning("⚠️ No data found. Ensure the Airflow pipeline has run and FastAPI is accessible.")
else:
    # Seniority Level via Checkboxes
    st.sidebar.markdown("**Seniority Level**")
    available_seniorities = list(all_data["level"].dropna().unique()) if "level" in all_data.columns else (
        list(all_data["seniority_level"].dropna().unique()) if "seniority_level" in all_data.columns else ["Unspecified"]
    )

    selected_seniority = []
    for sen in available_seniorities:
        if st.sidebar.checkbox(f"Seniority: {sen}", value=True, key=f"sen_{sen}"):
            selected_seniority.append(sen)

    st.sidebar.markdown("---")

    # Work Model via Checkboxes
    st.sidebar.markdown("**Work Model**")
    available_models = list(all_data["work_model"].dropna().unique()) if "work_model" in all_data.columns else ["Remote", "Hybrid", "On-site"]

    selected_work_models = []
    for model in available_models:
        if st.sidebar.checkbox(f"Model: {model}", value=True, key=f"mod_{model}"):
            selected_work_models.append(model)

    filtered_jobs = all_data.copy()

    sen_col = "seniority_level" if "seniority_level" in filtered_jobs.columns else ("level" if "level" in filtered_jobs.columns else None)
    if sen_col and selected_seniority:
        filtered_jobs = filtered_jobs[filtered_jobs[sen_col].isin(selected_seniority)]

    if "work_model" in filtered_jobs.columns and selected_work_models:
        filtered_jobs = filtered_jobs[filtered_jobs["work_model"].isin(selected_work_models)]

    st.title("German Tech Role Market Segmentation")
    st.caption("Combining statutory quartiles with live API insights.")

    salary_data = fetch_salary_data_from_fastapi()[0]["salaryData"]
    import os
    os.write(1, b'{ENTGELDATLAS_BASELINES.keys()}')
    print('salaryData:', salary_data)
    print('salaryData keys:', list(salary_data.keys()))
    print('selected_role:', selected_role)
    print('job keys: ', list(ENTGELDATLAS_BASELINES.keys()))
    salary_stats = {}
    

    if selected_role in list(ENTGELDATLAS_BASELINES.keys()):
        for key in list(ENTGELDATLAS_BASELINES.keys()):
            if selected_role in key:
                print('key: ', key)
                code = re.search(r'^[0-9]+$', key)
                salary_stats = salary_data[code]

        st.subheader(f"Statutory Salary Baseline: {selected_role}")
        kpis = salary_stats if salary_stats else ENTGELDATLAS_BASELINES[selected_role]
        col_p25, col_med, col_p75 = st.columns(3)
        col_p25.metric("25th Percentile", f"€{kpis['p25']:,}")
        col_med.metric("National Median", f"€{kpis['median']:,}")
        col_p75.metric("75th Percentile", f"€{kpis['p75']:,}")
        st.markdown("---")

    if len(filtered_jobs) < 3 or "cluster_id" not in filtered_jobs.columns:
        st.warning("Not enough job postings match your filters to render clustering visuals. Please broaden your search.")
    else:
        if "cluster_name" not in filtered_jobs.columns:
            filtered_jobs["cluster_name"] = filtered_jobs["cluster_id"].map(CLUSTER_LABELS)

        centroid_matrix = compute_centroids(filtered_jobs)

        tab_heatmap, tab_deepdive, tab_monitoring = st.tabs([
            "📊 Heatmap & Charts",
            "🔍 Archetype Deep Dive",
            "📈 ML Monitoring"
        ])

        with tab_heatmap:
            st.subheader("Discovered Market Archetypes & Visualizations")

            sub_chart_type, sub_volumes = st.tabs(["Skill Prevalence Heatmap", "Posting Distribution"])

            with sub_chart_type:
                if not centroid_matrix.empty:
                    top_skills = centroid_matrix.mean().nlargest(min(8, centroid_matrix.shape[1])).index
                    plot_matrix = centroid_matrix[top_skills]

                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    sns.heatmap(
                        plot_matrix,
                        cmap="YlGnBu",
                        annot=True,
                        fmt=".2f",
                        cbar_kws={"label": "Probability"},
                        ax=ax,
                        annot_kws={"size": 8},
                        linewidths=0.5
                    )
                    ax.set_ylabel("Cluster Archetype", fontsize=9)
                    ax.set_xlabel("Extracted Skills", fontsize=9)
                    plt.xticks(rotation=40, ha='right', fontsize=8)
                    plt.yticks(fontsize=8)
                    plt.subplots_adjust(bottom=0.32, left=0.28, right=0.95, top=0.92)
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Insufficient skill metrics for heatmap generation.")

            with sub_volumes:
                vol_series = filtered_jobs["cluster_name"].value_counts().sort_values(ascending=True)

                fig_vol, ax_vol = plt.subplots(figsize=(8, 3.5))
                vol_series.plot(kind="barh", ax=ax_vol, color="#2b5c8f")
                ax_vol.set_xlabel("Number of Job Postings", fontsize=9)
                ax_vol.set_ylabel("Cluster Archetype", fontsize=9)
                ax_vol.tick_params(axis='both', labelsize=8)
                plt.subplots_adjust(bottom=0.20, left=0.28, right=0.95, top=0.92)
                st.pyplot(fig_vol, clear_figure=True)

        with tab_deepdive:
            st.subheader("Archetype Deep Dive & Raw Postings")
            active_archetypes = list(filtered_jobs["cluster_name"].dropna().unique())

            if active_archetypes:
                selected_archetype = st.radio("Select Cluster to Inspect:", options=active_archetypes, horizontal=True)
                cluster_slice = filtered_jobs[filtered_jobs["cluster_name"] == selected_archetype]

                col_geo, col_tbl = st.columns([1, 2])
                with col_geo:
                    st.markdown("**Top Geographic Hubs**")
                    loc_col = "location_city" if "location_city" in cluster_slice.columns else "location"
                    if loc_col in cluster_slice.columns:
                        geo_counts = cluster_slice[loc_col].value_counts().reset_index()
                        geo_counts.columns = ["City", "Vacancies"]
                        st.dataframe(geo_counts, use_container_width=True, hide_index=True)
                    else:
                        st.write("Location data unavailable.")

                with col_tbl:
                    st.markdown("**Sample Postings in this Archetype**")
                    display_cols = [c for c in ["company_name", "job_title", "location_city", "work_model", "tech_stack"] if c in cluster_slice.columns]
                    st.dataframe(cluster_slice[display_cols].head(8), use_container_width=True, hide_index=True)

        with tab_monitoring:
            st.subheader("ML Model Performance & Drift Tracking")
            st.markdown("""
            ### What is happening under the hood?
            Your automated Airflow pipeline continuously ingests fresh job postings, extracts skill combinations using a dictionary-based NLP parser (**FlashText**), and groups the market landscape using an unsupervised clustering algorithm.

            This tab tracks the mathematical health, stability, and evolution of the clustering models across consecutive pipeline runs fetched from the FastAPI backend.

            ---
            ### Key Metrics Explained:
            * **Inertia (Within-Cluster Sum of Squares):** Measures how tightly grouped the clusters are. **Lower values are better**, indicating that job postings within the same archetype share very similar technology stacks.
            * **Silhouette Score:** Evaluates how similar a job posting is to its own cluster compared to other clusters. It ranges from `-1` to `+1`:
              * **Scores near +1** indicate dense, well-separated archetypes.
              * **Scores near 0 or negative** suggest overlapping skill requirements or poor separation.
            """)
            if st.button("Clear Cache & Reload Data"):
                st.cache_data.clear()  # Clears the cached API responses
                st.rerun()             # Instantly refreshes the dashboard

            metrics_df = fetch_metrics_from_fastapi()

            if metrics_df.empty:
                st.info("ℹ️ No historical metrics found. They will appear here once your Airflow clustering DAG executes its next run.")
            else:
                st.markdown("### 📋 Recent Model Runs Log")
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                if "execution_date" in metrics_df.columns and "inertia" in metrics_df.columns:
                    st.markdown("### 📉 Inertia Trend Over Time")
                    st.line_chart(metrics_df.set_index("execution_date")["inertia"])
