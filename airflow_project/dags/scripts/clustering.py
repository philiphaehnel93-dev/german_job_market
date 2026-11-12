import os
import json
import joblib
import pandas as pd
from pymongo import MongoClient, UpdateOne
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from google.cloud import storage

def run_kmeans_clustering_and_upsert_mongo(features_path: str, enriched_path: str, n_clusters: int = 5):
    print(f"📥 Loading features and enriched records...")
    features_df = pd.read_parquet(features_path)
    enriched_df = pd.read_parquet(enriched_path)

    if features_df.empty or enriched_df.empty:
        return

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
        # Use .predict() instead of .fit_predict() to avoid retraining
        enriched_df['cluster_id'] = kmeans.predict(features_df)
    else:
        print(f"🧠 No model found. Training initial model with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        enriched_df['cluster_id'] = kmeans.fit_predict(features_df)

        # Save the model locally so tomorrow's run uses it for inference
        joblib.dump(kmeans, model_path)
        print(f"💾 Model saved locally to {model_path}")

        # Upload the newly trained model back to GCS
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

    print("💾 Connecting directly to local MongoDB...")
    # Updated fallback to match your .env configuration
    mongo_uri = os.getenv("MONGO_URI", "mongodb://admin:admin@mongodb:27017/")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)

    # 1. Upsert the jobs
    collection = client["job_market_db"]["jobs_master"]
    operations = [UpdateOne({"origin_id": j["origin_id"]}, {"$set": j}, upsert=True) for j in records if "origin_id" in j]
    if operations:
        collection.bulk_write(operations)

    # 2. Upsert the ML Metrics
    metrics_collection = client["job_market_db"]["ml_metrics_history"]
    metrics_collection.insert_one({
        "execution_date": pd.Timestamp.utcnow(),
        "inertia": inertia,
        "silhouette_score": float(sil_score),
        "model_version": "v1.0"
    })

    print("✅ Upsert and Metrics logging complete!")
