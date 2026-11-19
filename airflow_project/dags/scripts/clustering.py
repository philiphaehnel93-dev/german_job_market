"""
K-Means Clustering & MongoDB Upsert Task
Uses a StandardScaler + PCA + KMeans Pipeline for consistent training and inference.
"""

import json
import os
from time import perf_counter

import joblib
import pandas as pd
import requests
from airflow.operators.python import get_current_context
from google.cloud import storage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "v2.0"
MODEL_PATH = "/app/clean_data/kmeans_production_model_v2.joblib"
GCS_BLOB_NAME = "models/kmeans_production_model_v2.joblib"
PROTECTED_COLS = {"is_remote", "tech_stack_size", "seniority_score"}


def _select_training_columns(features_df: pd.DataFrame) -> list:
    """Prunes rare tech columns adaptively during initial model training."""
    min_freq = max(2, int(0.02 * len(features_df)))
    keep_cols = [
        col
        for col in features_df.columns
        if col in PROTECTED_COLS
        or col.startswith("model_")
        or features_df[col].sum() >= min_freq
    ]

    # Fallback: if batch is very small and all tech columns were pruned, keep non-zero columns
    tech_kept = [c for c in keep_cols if c.startswith("tech_")]
    if not tech_kept:
        keep_cols = [
            col
            for col in features_df.columns
            if col in PROTECTED_COLS or features_df[col].sum() > 0
        ]

    return keep_cols if keep_cols else list(features_df.columns)


def run_kmeans_clustering_and_upsert_mongo(
    features_path: str,
    enriched_path: str,
    n_clusters: int = 5,
) -> None:
    start_time = perf_counter()
    print("📥 Loading features and enriched records...")
    features_df = pd.read_parquet(features_path)
    enriched_df = pd.read_parquet(enriched_path)

    if features_df.empty or enriched_df.empty:
        print("⚠️ Empty input dataframe. Skipping clustering.")
        return

    context = get_current_context()
    run_id = context["dag_run"].run_id
    task_id = context["task"].task_id

    bucket_name = os.getenv("AIRFLOW_VAR_GCP_BUCKET_NAME", "ml-pipeline-hu1tac-ml-data")
    blob = None

    # --- GCS DOWNLOAD LOGIC ---
    print("☁️ Checking GCS for existing v2 model pipeline...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(GCS_BLOB_NAME)

        if blob.exists():
            print("☁️ Model found in GCS! Downloading to local volume...")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            blob.download_to_filename(MODEL_PATH)
        else:
            print("☁️ No v2 model found in GCS yet.")
    except Exception as exc:
        print(f"⚠️ Could not connect to GCS: {exc}")

    # --- TRAINING / INFERENCE LOGIC ---
    model_pipeline = None
    inertia = 0.0
    sil_score = 0.0

    if os.path.exists(MODEL_PATH):
        print("🔍 Local v2 pipeline found! Loading for inference...")
        model_pipeline = joblib.load(MODEL_PATH)
        expected_features = model_pipeline.feature_names_in_
        features_df = features_df.reindex(columns=expected_features, fill_value=0)
        enriched_df["cluster_id"] = model_pipeline.predict(features_df)
    else:
        n_samples = len(features_df)
        if n_samples < 2:
            print(
                f"⚠️ Only {n_samples} row(s) available and no trained model exists. "
                "Assigning cluster_id=0 without persisting a model."
            )
            enriched_df["cluster_id"] = 0
        else:
            effective_k = min(n_clusters, n_samples)
            if effective_k < n_clusters:
                print(
                    f"⚠️ Batch has only {n_samples} rows (< n_clusters={n_clusters}). "
                    f"Using temporary k={effective_k} and skipping model persistence."
                )

            keep_cols = _select_training_columns(features_df)
            features_df = features_df[keep_cols]
            print(
                f"🧠 Training Pipeline (StandardScaler + PCA + KMeans k={effective_k}) "
                f"on shape {features_df.shape}..."
            )

            steps = [("scaler", StandardScaler())]
            if features_df.shape[1] > 1 and n_samples > 2:
                steps.append(("pca", PCA(n_components=0.90, random_state=42)))
            steps.append(("kmeans", KMeans(n_clusters=effective_k, random_state=42, n_init=10)))

            model_pipeline = Pipeline(steps)
            enriched_df["cluster_id"] = model_pipeline.fit_predict(features_df)

            # Only persist production model when trained on a full-size batch
            if effective_k == n_clusters:
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                joblib.dump(model_pipeline, MODEL_PATH)
                print(f"💾 Pipeline saved locally to {MODEL_PATH}")

                if blob is not None:
                    try:
                        print("☁️ Uploading new v2 pipeline to GCS...")
                        blob.upload_from_filename(MODEL_PATH)
                        print("✅ Upload to GCS complete.")
                    except Exception as exc:
                        print(f"⚠️ Failed to upload model to GCS: {exc}")

    # --- CALCULATE PERFORMANCE METRICS ---
    if model_pipeline is not None:
        kmeans_step = model_pipeline.named_steps["kmeans"]
        inertia = float(kmeans_step.inertia_) if hasattr(kmeans_step, "inertia_") else 0.0

        # Evaluate Silhouette Score in the exact scaled/PCA space clustered by KMeans
        transformed_features = model_pipeline[:-1].transform(features_df)
        unique_clusters = enriched_df["cluster_id"].nunique()
        if 1 < unique_clusters < len(enriched_df):
            sil_score = float(silhouette_score(transformed_features, enriched_df["cluster_id"]))
        else:
            print(
                f"⚠️ Cannot compute silhouette score with {unique_clusters} unique cluster(s) "
                f"across {len(enriched_df)} row(s). Defaulting to 0.0."
            )
            sil_score = 0.0

    print(f"📊 Model Metrics - Inertia: {inertia:.2f}, Silhouette: {sil_score:.4f}")

    # --- UPSERT RECORDS TO MONGODB VIA FASTAPI ---
    records = json.loads(enriched_df.to_json(orient="records"))
    for record in records:
        record["ml_model_version"] = MODEL_VERSION

    print("💾 Connecting to FastAPI to bulk insert to local MongoDB...")
    fastapi_url = "http://airflow-fastapi:8081/api/v1/jobs/bulk"
    try:
        response = requests.post(fastapi_url, json=records, timeout=15)
        response.raise_for_status()
        print(f"✅ Successfully saved via FastAPI: {response.json()}")
    except requests.exceptions.RequestException as exc:
        print(f"⚠️ FastAPI Error: {exc}")
        raise

    # --- LOG ML MODEL METRICS ---
    metrics_payload = {
        "execution_date": str(pd.Timestamp.utcnow()),
        "inertia": inertia,
        "silhouette_score": sil_score,
        "model_version": MODEL_VERSION,
        "dag_run_id": run_id,
        "task_id": task_id,
    }

    print("📊 Sending ML model metrics to FastAPI...")
    try:
        token = os.getenv("API_SECRET_KEY")
        headers = {"Authorization": f"Bearer {token}"}
        metrics_response = requests.post(
            "http://airflow-fastapi:8081/api/v1/ml_metrics",
            json=metrics_payload,
            headers=headers,
            timeout=10,
        )
        metrics_response.raise_for_status()
        print("✅ ML metrics successfully logged!")
    except requests.exceptions.RequestException as exc:
        print(f"⚠️ Failed to log ML metrics: {exc}")

    # --- LOG PIPELINE EXECUTION METRICS ---
    actual_duration = round(perf_counter() - start_time, 2)
    print("📡 Sending pipeline execution metrics to FastAPI...")
    try:
        pipeline_response = requests.post(
            "http://airflow-fastapi:8081/api/v1/pipeline_metrics",
            json={
                "dag_id": "german_job_market_pipeline",
                "run_id": run_id,
                "task_id": task_id,
                "status": "success",
                "rows_processed": len(records),
                "duration_seconds": actual_duration,
                "execution_date": str(pd.Timestamp.utcnow()),
            },
            timeout=5,
        )
        pipeline_response.raise_for_status()
        print("✅ Pipeline metrics successfully logged!")
    except requests.exceptions.RequestException as exc:
        print(f"❌ Failed to log pipeline metrics: {exc}")
        raise

    print("✅ All tasks complete!")
