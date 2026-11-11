import os
import json
import pandas as pd
from pymongo import MongoClient, UpdateOne
from sklearn.cluster import KMeans

def run_kmeans_clustering_and_upsert_mongo(features_path: str, enriched_path: str, n_clusters: int = 5):
    """Runs K-Means clustering and performs a direct UPSERT to the local MongoDB container."""
    print(f"📥 Loading features from {features_path} and enriched records from {enriched_path}...")

    features_df = pd.read_parquet(features_path)
    enriched_df = pd.read_parquet(enriched_path)

    if features_df.empty or enriched_df.empty:
        print("⚠️ No data available for clustering. Skipping.")
        return

    print(f"🧠 Training K-Means model with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    enriched_df['cluster_id'] = kmeans.fit_predict(features_df)

    records = json.loads(enriched_df.to_json(orient="records"))

    print("💾 Connecting directly to local MongoDB...")
    # 'local_mongo_db' is the container name on your Docker network
    mongo_uri = os.getenv("MONGO_URI", "mongodb://admin:admin@local_mongo_db:27017/")

    # serverSelectionTimeoutMS=5000 forces PyMongo to throw an error after 5 seconds if it can't find the DB
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client["job_market_db"]
    collection = db["jobs_master"]

    operations = []
    for job in records:
        if "origin_id" in job:
            operations.append(
                UpdateOne({"origin_id": job["origin_id"]}, {"$set": job}, upsert=True)
            )

    if operations:
        print("🚀 Executing bulk upsert...")
        result = collection.bulk_write(operations)
        print(f"✅ Upsert complete! Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted: {result.upserted_count}")
    else:
        print("⚠️ No valid operations to perform.")
