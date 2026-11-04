"""
Market Segmentation & National Baseline Dashboard
Analyzes German Tech Roles using Entgeldatlas Baselines and MongoDB Clustered Data.
"""

from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import streamlit as st

# -----------------------------------------------------------------------------
# DATA SOURCES & SCHEMAS
# -----------------------------------------------------------------------------

ENTGELDATLAS_BASELINES: Dict[str, Dict[str, int]] = {
    "Data Engineer (43214)": {"p25": 55_200, "median": 72_400, "p75": 88_900},
    "Data Scientist (43104)": {"p25": 58_100, "median": 75_600, "p75": 92_300},
    "Software Engineer (43414)": {"p25": 52_400, "median": 68_800, "p75": 85_100},
}

CLUSTER_LABELS: Dict[int, str] = {
    0: "Cloud & Infrastructure DE",
    1: "Business Intelligence & SQL Analytics",
    2: "Machine Learning & AI Engineering",
    3: "Big Data & Distributed Streaming",
    4: "Modern Data Platform & Warehousing",
    5: "Backend Data Services",
    6: "Core Python/DevOps Generalist",
}

@st.cache_data(ttl=600)
def load_job_data_from_mongo() -> pd.DataFrame:
    """Fetches real clustered job data directly from MongoDB Atlas."""
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        st.error("Error: MONGO_URI environment variable is not set.")
        return pd.DataFrame()

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client["job_market_db"]
        collection = db["jobs_clustered"]
        
        records = list(collection.find({}, {"_id": 0}))
        if not records:
            return pd.DataFrame()
            
        df = pd.DataFrame(records)
        
        # Filter out rows where tech_stack is missing or is an empty list
        if "tech_stack" in df.columns:
            df = df[df["tech_stack"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        
        if "location" in df.columns and "location_city" not in df.columns:
            df["location_city"] = df["location"]
            
        return df
    except ConnectionFailure:
        st.error("Error: Failed to connect to MongoDB Atlas.")
        return pd.DataFrame()


def compute_centroids_from_db(df: pd.DataFrame) -> pd.DataFrame:
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
selected_role = st.sidebar.selectbox("Occupational Target (KldB)", list(ENTGELDATLAS_BASELINES.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("Job Postings Filters")

all_data = load_job_data_from_mongo()

if all_data.empty:
    st.warning("⚠️ No data found in MongoDB `jobs_clustered`. Run your Airflow pipeline to populate the database.")
else:
    available_seniorities = list(all_data["seniority_level"].unique()) if "seniority_level" in all_data.columns else ["Unspecified"]
    selected_seniority = st.sidebar.multiselect("Seniority Level", options=available_seniorities, default=available_seniorities)

    available_models = list(all_data["work_model"].unique()) if "work_model" in all_data.columns else ["Remote", "Hybrid", "On-site"]
    selected_work_models = st.sidebar.multiselect("Work Model", options=available_models, default=available_models)

    filtered_jobs = all_data.copy()
    if "seniority_level" in filtered_jobs.columns and selected_seniority:
        filtered_jobs = filtered_jobs[filtered_jobs["seniority_level"].isin(selected_seniority)]
    if "work_model" in filtered_jobs.columns and selected_work_models:
        filtered_jobs = filtered_jobs[filtered_jobs["work_model"].isin(selected_work_models)]

    st.title("German Tech Role Market Segmentation")
    st.caption("Combining Bundesagentur für Arbeit statutory quartiles with live MongoDB insights.")

    st.subheader(f"Statutory Salary Baseline: {selected_role}")
    kpis = ENTGELDATLAS_BASELINES[selected_role]
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

        centroid_matrix = compute_centroids_from_db(filtered_jobs)

        st.subheader("Discovered Market Archetypes")
        tab_heatmap, tab_volumes = st.tabs(["Skill Prevalence Heatmap", "Posting Distribution"])

        with tab_heatmap:
            if not centroid_matrix.empty:
                top_skills = centroid_matrix.mean().nlargest(min(15, centroid_matrix.shape[1])).index
                plot_matrix = centroid_matrix[top_skills]

                fig, ax = plt.subplots(figsize=(12, 5))
                sns.heatmap(plot_matrix, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={"label": "Probability"}, ax=ax)
                ax.set_ylabel("Cluster Archetype")
                ax.set_xlabel("Extracted Skills")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Insufficient skill metrics for heatmap generation.")

        with tab_volumes:
            vol_series = filtered_jobs["cluster_name"].value_counts().sort_values(ascending=True)
            fig_vol, ax_vol = plt.subplots(figsize=(10, 4))
            vol_series.plot(kind="barh", ax=ax_vol, color="#2b5c8f")
            ax_vol.set_xlabel("Number of Job Postings")
            ax_vol.set_ylabel("")
            st.pyplot(fig_vol, clear_figure=True)

        st.markdown("---")
        st.subheader("Archetype Deep Dive & Raw Postings")

        active_archetypes = list(filtered_jobs["cluster_name"].dropna().unique())
        if active_archetypes:
            selected_archetype = st.selectbox("Select Cluster to Inspect:", options=active_archetypes)
            cluster_slice = filtered_jobs[filtered_jobs["cluster_name"] == selected_archetype]

            col_geo, col_tbl = st.columns([1, 2])
            with col_geo:
                st.markdown("**Top Geographic Hubs (Postings Count)**")
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
