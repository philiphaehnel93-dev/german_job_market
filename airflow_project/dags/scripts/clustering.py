"""
K-Means Clustering & MongoDB Upsert Task
Combines historical MongoDB jobs with new incremental records to train a
StandardScaler + PCA + KMeans pipeline, scaling cleanly past 10,000 rows.
"""

import json
import os
from time import perf_counter
from typing import Tuple

import joblib
import pandas as pd
import requests
from airflow.operators.python import get_current_context
from google.cloud import storage
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from scripts.ml_preprocessing import SENIORITY_MAP, clean_tech_list, extract_is_remote

MODEL_VERSION = os.getenv("ML_MODEL_VERSION", "v2.0")
MODEL_PATH = os.getenv(
    "ML_MODEL_PATH", f"/app/clean_data/kmeans_production_model_{MODEL_VERSION}.joblib"
)
GCS_BLOB_NAME = os.getenv(
    "ML_GCS_BLOB_NAME", f"models/kmeans_production_model_{MODEL_VERSION}.joblib"
)
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://airflow-fastapi:8081/api/v1")

PROTECTED_COLS = {"is_remote", "tech_stack_size", "seniority_score"}
MAX_TRAIN_ROWS = 10_000
SILHOUETTE_SAMPLE_SIZE = 5_000


def _get_auth_headers() -> dict:
    token = os.getenv("API_SECRET_KEY", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _fetch_historical_jobs(limit: int = MAX_TRAIN_ROWS) -> pd.DataFrame:
    """Fetches historical jobs from FastAPI so clustering reflects the full dataset."""
    url = f"{FASTAPI_BASE_URL}/jobs"
    try:
        response = requests.get(
            url,
            params={"limit": limit, "include_description": False},
            headers=_get_auth_headers(),
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        records = payload.get("data", []) if isinstance(payload, dict) else payload
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        print(f"📚 Loaded {len(df)} historical jobs from FastAPI.")
        return df
    except requests.exceptions.RequestException as exc:
        print(f"⚠️ Could not fetch historical jobs ({exc}). Falling back to current batch only.")
        return pd.DataFrame()


def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Builds a unified numeric feature matrix from combined historical + new job records."""
    is_remote_series = extract_is_remote(df)

    if "tech_stack" in df.columns:
        clean_stacks = df["tech_stack"].apply(clean_tech_list)
    else:
        clean_stacks = pd.Series([[] for _ in range(len(df))], index=df.index)

    mlb = MultiLabelBinarizer()
    tech_matrix = mlb.fit_transform(clean_stacks)
    tech_dummies = pd.DataFrame(
        tech_matrix,
        columns=[f"tech_{col}" for col in mlb.classes_],
        index=df.index,
    )

    feature_parts = [
        pd.DataFrame(
            {
                "is_remote": is_remote_series,
                "tech_stack_size": clean_stacks.apply(len),
            },
            index=df.index,
        ),
        tech_dummies,
    ]

    sen_col = next((c for c in ("seniority_level", "level") if c in df.columns), None)
    if sen_col:
        seniority_score = df[sen_col].map(SENIORITY_MAP).fillna(2).astype(int)
        feature_parts.append(pd.DataFrame({"seniority_score": seniority_score}, index=df.index))

    if "work_model" in df.columns:
        work_model_dummies = pd.get_dummies(
            df["work_model"].fillna("Unspecified").astype(str),
            prefix="model",
            dtype=int,
        )
        feature_parts.append(work_model_dummies)

    feature_df = pd.concat(feature_parts, axis=1).fillna(0)
    feature_df.columns = [str(col) for col in feature_df.columns]
    return feature_df


def _select_training_columns(features_df: pd.DataFrame) -> list:
    """Prunes rare tech columns adaptively during model training."""
    min_freq = max(2, int(0.02 * len(features_df)))
    keep_cols = [
        col
        for col in features_df.columns
        if col in PROTECTED_COLS
        or col.startswith("model_")
        or features_df[col].sum() >= min_freq
    ]

    tech_kept = [c for c in keep_cols if c.startswith("tech_")]
    if not tech_kept:
        keep_cols = [
            col
            for col in features_df.columns
            if col in PROTECTED_COLS or features_df[col].sum() > 0
        ]

    return keep_cols if keep_cols else list(features_df.columns)


def _train_pipeline(
    features_df: pd.DataFrame, n_clusters: int
) -> Tuple[Pipeline, pd.Series]:
    """Trains StandardScaler + PCA + KMeans (or MiniBatchKMeans for >10k rows)."""
    n_samples = len(features_df)
    effective_k = min(n_clusters, n_samples)

    steps = [("scaler", StandardScaler())]
    if features_df.shape[1] > 1 and n_samples > 2:
        steps.append(("pca", PCA(n_components=0.90, random_state=42)))

    if n_samples > MAX_TRAIN_ROWS:
        print(
            f"⚡ Dataset has {n_samples} rows (> {MAX_TRAIN_ROWS}). "
            f"Using MiniBatchKMeans (k={effective_k}) for scalability."
        )
        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            random_state=42,
            batch_size=1024,
            n_init=10,
        )
    else:
        kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init=10)

    steps.append(("kmeans", kmeans))
    pipeline = Pipeline(steps)
    labels = pd.Series(pipeline.fit_predict(features_df), index=features_df.index)
    return pipeline, labels


def run_kmeans_clustering_and_upsert_mongo(
    features_path: str,
    enriched_path: str,
    n_clusters: int = 5,
) -> None:
    start_time = perf_counter()
    print("📥 Loading new enriched records...")
    new_enriched_df = (
        pd.read_parquet(enriched_path)
        if os.path.exists(enriched_path)
        else pd.DataFrame()
    )

    context = get_current_context()
    run_id = context["dag_run"].run_id
    task_id = context["task"].task_id

    # --- COMBINE HISTORICAL + NEW DATA ---
    historical_df = _fetch_historical_jobs(limit=MAX_TRAIN_ROWS)

    if not historical_df.empty and not new_enriched_df.empty:
        combined_df = pd.concat([historical_df, new_enriched_df], ignore_index=True)
    elif not historical_df.empty:
        combined_df = historical_df.copy()
    elif not new_enriched_df.empty:
        combined_df = new_enriched_df.copy()
    else:
        print("⚠️ No historical or new jobs available. Skipping clustering.")
        return

    if "origin_id" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["origin_id"], keep="last")
    if len(combined_df) > MAX_TRAIN_ROWS:
        combined_df = combined_df.tail(MAX_TRAIN_ROWS).reset_index(drop=True)
    else:
        combined_df = combined_df.reset_index(drop=True)

    print(f"🧩 Building unified feature matrix on {len(combined_df)} total jobs...")
    combined_features = _build_feature_matrix(combined_df)
    keep_cols = _select_training_columns(combined_features)
    combined_features = combined_features[keep_cols]

    # --- TRAIN ON COMBINED DATASET ---
    n_rows = len(combined_features)
    model_pipeline = None
    inertia = 0.0
    sil_score = None
    unique_clusters = 1

    if n_rows < 2:
        print(f"⚠️ Only {n_rows} row(s) total. Assigning cluster_id=0.")
        if not new_enriched_df.empty:
            new_enriched_df["cluster_id"] = 0
    else:
        print(
            f"🧠 Training {MODEL_VERSION} pipeline on {combined_features.shape} "
            f"(k={min(n_clusters, n_rows)})..."
        )
        model_pipeline, combined_labels = _train_pipeline(combined_features, n_clusters)

        # Predict cluster_id only if there are new records to upsert
        if not new_enriched_df.empty:
            new_features = _build_feature_matrix(new_enriched_df).reindex(
                columns=keep_cols, fill_value=0
            )
            new_enriched_df["cluster_id"] = model_pipeline.predict(new_features)

        # --- CALCULATE PERFORMANCE METRICS ON COMBINED DATASET ---
        kmeans_step = model_pipeline.named_steps["kmeans"]
        inertia = float(kmeans_step.inertia_) if hasattr(kmeans_step, "inertia_") else 0.0
        unique_clusters = int(combined_labels.nunique())

        transformed_features = model_pipeline[:-1].transform(combined_features)
        if 1 < unique_clusters < n_rows:
            sample_sz = SILHOUETTE_SAMPLE_SIZE if n_rows > SILHOUETTE_SAMPLE_SIZE else None
            sil_score = float(
                silhouette_score(
                    transformed_features,
                    combined_labels,
                    sample_size=sample_sz,
                    random_state=42,
                )
            )
        else:
            print(
                f"⚠️ Cannot compute silhouette score with {unique_clusters} unique cluster(s) "
                f"across {n_rows} row(s)."
            )

        # Persist model locally and to GCS when trained on a full-size corpus
        if n_rows >= n_clusters:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump(model_pipeline, MODEL_PATH)
            print(f"💾 Pipeline saved locally to {MODEL_PATH}")

            bucket_name = os.getenv(
                "AIRFLOW_VAR_GCP_BUCKET_NAME", "ml-pipeline-hu1tac-ml-data"
            )
            try:
                storage_client = storage.Client()
                blob = storage_client.bucket(bucket_name).blob(GCS_BLOB_NAME)
                blob.upload_from_filename(MODEL_PATH)
                print(f"☁️ Uploaded {MODEL_VERSION} pipeline to GCS.")
            except Exception as exc:
                print(f"⚠️ Failed to upload model to GCS: {exc}")

    sil_display = f"{sil_score:.4f}" if sil_score is not None else "N/A"
    print(
        f"📊 Model Metrics - Corpus Rows: {n_rows}, Clusters: {unique_clusters}, "
        f"Inertia: {inertia:.2f}, Silhouette: {sil_display}"
    )

    # --- UPSERT NEW RECORDS TO MONGODB VIA FASTAPI ---
    records = []
    if not new_enriched_df.empty:
        records = json.loads(new_enriched_df.to_json(orient="records"))
        for record in records:
            record["ml_model_version"] = MODEL_VERSION

        print(f"💾 Bulk inserting {len(records)} new jobs via FastAPI...")
        try:
            response = requests.post(
                f"{FASTAPI_BASE_URL}/jobs/bulk",
                json=records,
                headers=_get_auth_headers(),
                timeout=15,
            )
            response.raise_for_status()
            print(f"✅ Successfully saved via FastAPI: {response.json()}")
        except requests.exceptions.RequestException as exc:
            print(f"⚠️ FastAPI Error: {exc}")
            raise
    else:
        print("ℹ️ No new jobs to upsert; proceeding to log model metrics.")

    # --- LOG ML MODEL METRICS (only when silhouette is valid) ---
    if sil_score is not None:
        metrics_payload = {
            "execution_date": str(pd.Timestamp.utcnow()),
            "inertia": inertia,
            "silhouette_score": sil_score,
            "n_rows": n_rows,
            "n_unique_clusters": unique_clusters,
            "model_version": MODEL_VERSION,
            "dag_run_id": run_id,
            "task_id": task_id,
        }
        print("📊 Sending ML model metrics to FastAPI...")
        try:
            metrics_response = requests.post(
                f"{FASTAPI_BASE_URL}/ml_metrics",
                json=metrics_payload,
                headers=_get_auth_headers(),
                timeout=10,
            )
            metrics_response.raise_for_status()
            print("✅ ML metrics successfully logged!")
        except requests.exceptions.RequestException as exc:
            print(f"⚠️ Failed to log ML metrics: {exc}")
    else:
        print("ℹ️ Skipping /ml_metrics log because silhouette score could not be computed.")

    # --- LOG PIPELINE EXECUTION METRICS ---
    actual_duration = round(perf_counter() - start_time, 2)
    print("📡 Sending pipeline execution metrics to FastAPI...")
    try:
        pipeline_response = requests.post(
            f"{FASTAPI_BASE_URL}/pipeline_metrics",
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
