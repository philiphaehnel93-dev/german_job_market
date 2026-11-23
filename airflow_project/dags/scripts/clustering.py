"""
K-Means Clustering & MongoDB Upsert Task (Inference + Drift Monitoring)
- Loads the frozen production pipeline from GCS/local volume for consistent inference.
- Predicts cluster_id on new incremental jobs and upserts them to MongoDB via FastAPI.
- Evaluates silhouette score on the combined corpus (historical + new) to monitor drift.
- Only trains and uploads a new model if no model exists for the configured MODEL_VERSION.
"""

import json
import os
from time import perf_counter
from typing import Optional, Tuple

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

try:
    from ml_preprocessing import SENIORITY_MAP, clean_tech_list, extract_is_remote
except ImportError:
    from scripts.ml_preprocessing import SENIORITY_MAP, clean_tech_list, extract_is_remote

FASTAPI_BASE_URL = "http://airflow-fastapi:8081/api/v1"
MODEL_VERSION = "v2.0"
MODEL_PATH = "/app/clean_data/kmeans_production_model_v2.joblib"
GCS_BLOB_NAME = "models/kmeans_production_model_v2.joblib"
PROTECTED_COLS = {"is_remote", "tech_stack_size", "seniority_score"}
MAX_EVAL_ROWS = 10_000
SILHOUETTE_SAMPLE_SIZE = 5_000


def _get_auth_headers() -> dict:
    token = os.getenv("API_SECRET_KEY", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _fetch_historical_jobs(limit: int = MAX_EVAL_ROWS) -> pd.DataFrame:
    """Fetches historical jobs from FastAPI (excluding descriptions) for corpus-level evaluation."""
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


def _download_model_from_gcs(bucket_name: str) -> Tuple[Optional[storage.Blob], bool]:
    """Checks GCS for the frozen model pipeline and downloads it locally if present."""
    print(f"☁️ Checking GCS for existing {MODEL_VERSION} model pipeline...")
    try:
        storage_client = storage.Client()
        blob = storage_client.bucket(bucket_name).blob(GCS_BLOB_NAME)
        if blob.exists():
            print("☁️ Model found in GCS! Downloading to local volume...")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            blob.download_to_filename(MODEL_PATH)
            return blob, True
        print(f"☁️ No {MODEL_VERSION} model found in GCS yet.")
        return blob, False
    except Exception as exc:
        print(f"⚠️ Could not connect to GCS: {exc}")
        return None, False


def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Builds a unified numeric feature matrix from job records."""
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


def _train_pipeline(
    features_df: pd.DataFrame, n_clusters: int
) -> Tuple[Pipeline, pd.Series]:
    """Trains StandardScaler + PCA + KMeans (or MiniBatchKMeans for >10k rows)."""
    n_samples = len(features_df)
    effective_k = min(n_clusters, n_samples)

    steps = [("scaler", StandardScaler())]
    if features_df.shape[1] > 1 and n_samples > 2:
        steps.append(("pca", PCA(n_components=0.90, random_state=42)))

    if n_samples > MAX_EVAL_ROWS:
        print(
            f"⚡ Dataset has {n_samples} rows (> {MAX_EVAL_ROWS}). "
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
        if enriched_path and os.path.exists(enriched_path)
        else pd.DataFrame()
    )

    context = get_current_context()
    run_id = context["dag_run"].run_id
    task_id = context["task"].task_id
    bucket_name = os.getenv("AIRFLOW_VAR_GCP_BUCKET_NAME", "ml-pipeline-hu1tac-ml-data")

    # --- BUILD COMBINED CORPUS FOR EVALUATION / BOOTSTRAP ---
    historical_df = _fetch_historical_jobs(limit=MAX_EVAL_ROWS)

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
    if len(combined_df) > MAX_EVAL_ROWS:
        combined_df = combined_df.tail(MAX_EVAL_ROWS).reset_index(drop=True)
    else:
        combined_df = combined_df.reset_index(drop=True)

    print(f"🧩 Building unified feature matrix on {len(combined_df)} total jobs...")
    combined_features = _build_feature_matrix(combined_df)

    # --- LOAD FROZEN MODEL FROM GCS/LOCAL, OR TRAIN IF MISSING ---
    blob, _ = _download_model_from_gcs(bucket_name)

    n_rows = len(combined_features)
    model_pipeline: Optional[Pipeline] = None
    combined_labels: Optional[pd.Series] = None
    eval_features: Optional[pd.DataFrame] = None
    inertia = 0.0
    sil_score = 0.0
    unique_clusters = 1

    if os.path.exists(MODEL_PATH):
        print(f"🔍 Frozen {MODEL_VERSION} pipeline found! Loading for inference & drift scoring...")
        model_pipeline = joblib.load(MODEL_PATH)
        expected_features = list(model_pipeline.feature_names_in_)

        missing_cols = [c for c in expected_features if c not in combined_features.columns]
        if missing_cols:
            print(
                f"⚠️ {len(missing_cols)}/{len(expected_features)} expected model features "
                "missing in corpus; filling with 0."
            )

        eval_features = combined_features.reindex(columns=expected_features, fill_value=0)
        combined_labels = pd.Series(
            model_pipeline.predict(eval_features), index=eval_features.index
        )

        # Predict cluster_id for new records only
        if not new_enriched_df.empty:
            new_features = _build_feature_matrix(new_enriched_df).reindex(
                columns=expected_features, fill_value=0
            )
            new_enriched_df["cluster_id"] = model_pipeline.predict(new_features)
    else:
        # Cold-start bootstrap: train and upload a new model for this MODEL_VERSION
        if n_rows < 2:
            print(f"⚠️ Only {n_rows} row(s) total and no trained model exists. Assigning cluster_id=0.")
            if not new_enriched_df.empty:
                new_enriched_df["cluster_id"] = 0
        else:
            keep_cols = _select_training_columns(combined_features)
            eval_features = combined_features[keep_cols]
            print(
                f"🧠 No existing {MODEL_VERSION} model found. Training new pipeline on "
                f"{eval_features.shape} (k={min(n_clusters, n_rows)})..."
            )
            model_pipeline, combined_labels = _train_pipeline(eval_features, n_clusters)

            if not new_enriched_df.empty:
                new_features = _build_feature_matrix(new_enriched_df).reindex(
                    columns=keep_cols, fill_value=0
                )
                new_enriched_df["cluster_id"] = model_pipeline.predict(new_features)

            if n_rows >= n_clusters:
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                joblib.dump(model_pipeline, MODEL_PATH)
                print(f"💾 Pipeline saved locally to {MODEL_PATH}")

                if blob is not None:
                    try:
                        blob.upload_from_filename(MODEL_PATH)
                        print(f"☁️ Uploaded new {MODEL_VERSION} pipeline to GCS.")
                    except Exception as exc:
                        print(f"⚠️ Failed to upload model to GCS: {exc}")

    # --- CALCULATE PERFORMANCE / DRIFT METRICS ON COMBINED CORPUS ---
    if model_pipeline is not None and combined_labels is not None and eval_features is not None:
        kmeans_step = model_pipeline.named_steps["kmeans"]
        inertia = float(kmeans_step.inertia_) if hasattr(kmeans_step, "inertia_") else 0.0
        unique_clusters = int(combined_labels.nunique())

        transformed_features = model_pipeline[:-1].transform(eval_features)
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
                f"across {n_rows} row(s). Defaulting to 0.0."
            )
            sil_score = 0.0

    print(
        f"📊 Model Metrics ({MODEL_VERSION}) - Eval Corpus Rows: {n_rows}, "
        f"Clusters: {unique_clusters}, Inertia: {inertia:.2f}, Silhouette: {sil_score:.4f}"
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
        print("ℹ️ No new jobs to upsert; proceeding to log drift/model metrics.")

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
            headers=_get_auth_headers(),
            timeout=5,
        )
        pipeline_response.raise_for_status()
        print("✅ Pipeline metrics successfully logged!")
    except requests.exceptions.RequestException as exc:
        print(f"❌ Failed to log pipeline metrics: {exc}")
        raise

    print("✅ All tasks complete!")
