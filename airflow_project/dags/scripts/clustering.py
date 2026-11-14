import os
import json
import joblib
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from google.cloud import storage
from airflow.operators.python import get_current_context

def run_kmeans_clustering_and_upsert_mongo(features_path: str, enriched_path: str, n_clusters: int = 5):
    print(f"📥 Loading features and enriched records...")
    features_df = pd.read_parquet(features_path)
    enriched_df = pd.read_parquet(enriched_path)

    if features_df.empty or enriched_df.empty:
        return

    # Get Airflow context early so we can use it for both ML and Pipeline metrics
    context = get_current_context()
    run_id = context['dag_run'].run_id
    task_id = context['task'].task_id

    # The trained model will be stored on the shared Docker volume
    model_path = "/app/clean_data/kmeans_production_model.joblib"

    bucket_name = os.getenv("AIRFLOW_VAR_GCP_BUCKET_NAME", "ml-pipeline-hu1tac-ml-data")
    blob_name = "models/kmeans_production_model.joblib"

    # --- GCS DOWNLOAD LOGIC ---
    print("☁️ Checking GCS for existing model...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if blob.exists():
            print("☁️ Model found in GCS! Downloading to local volume...")
            blob.download_to_filename(model_path)
        else:
            print("☁️ No model found in GCS yet.")
    except Exception as e:
        print(f"⚠️ Could not connect to GCS: {e}")

    # --- THE INFERENCE LOGIC ---
    if os.path.exists(model_path):
        print("🔍 Local model found! Loading for inference...")
        kmeans = joblib.load(model_path)
        enriched_df['cluster_id'] = kmeans.predict(features_df)
    else:
        print(f"🧠 No model found. Training initial model with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        enriched_df['cluster_id'] = kmeans.fit_predict(features_df)

        joblib.dump(kmeans, model_path)
        print(f"💾 Model saved locally to {model_path}")

        try:
            print("☁️ Uploading new model to GCS...")
            blob.upload_from_filename(model_path)
            print("✅ Upload to GCS complete.")
        except Exception as e:
            print(f"⚠️ Failed to upload model to GCS: {e}")

    # --- CALCULATE PERFORMANCE METRICS ---
    inertia = kmeans.inertia_
    sil_score = silhouette_score(features_df, enriched_df['cluster_id'])
    print(f"📊 Model Metrics - Inertia: {inertia:.2f}, Silhouette: {sil_score:.4f}")

    # Prepare data for MongoDB
    records = json.loads(enriched_df.to_json(orient="records"))
    for record in records:
        record["ml_model_version"] = "v1.2"

    print("💾 Connecting to FastAPI to bulk insert to local MongoDB...")
    fastapi_url = "http://airflow-fastapi:8081/api/v1/jobs/bulk"

    try:
        response = requests.post(fastapi_url, json=records, timeout=15)
        response.raise_for_status()
        print(f"✅ Successfully saved via FastAPI: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ FastAPI Error: {e}")
        raise

    metrics_payload = {
        "execution_date": str(pd.Timestamp.utcnow()),
        "inertia": float(inertia),
        "silhouette_score": float(sil_score),
        "model_version": "v1.2",
        "dag_run_id": run_id,
        "task_id": task_id
    }

    # --- Send Standard Pipeline Metrics ---
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
                "duration_seconds": 0.0,
                "execution_date": str(pd.Timestamp.utcnow())
            },
            timeout=5
        )
        pipeline_response.raise_for_status()
        print("✅ Pipeline metrics successfully logged!")
    except Exception as e:
        print(f"❌ Failed to log pipeline metrics: {e}")
        raise

    print("✅ All tasks complete!")
