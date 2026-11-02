import os
import pandas as pd
from sklearn.cluster import KMeans
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def run_kmeans_clustering(n_clusters: int = 5):
    """Trains a K-Means model on the feature matrix and assigns cluster labels."""
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise ValueError("Error: MONGO_URI environment variable not set.")

    db_name = "job_market_db"
    feature_collection = "jobs_features"
    enriched_collection = "jobs_enriched"
    target_collection = "jobs_clustered"

    print("🚀 Connecting to MongoDB for K-Means clustering...")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
    except ConnectionFailure as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    db = client[db_name]
    
    # Fetch feature vectors and original enriched job documents
    features = list(db[feature_collection].find({}, {"_id": 0}))
    enriched_jobs = list(db[enriched_collection].find({}, {"_id": 0}))

    if not features or not enriched_jobs:
        print("⚠️ Missing feature records or enriched jobs for clustering.")
        return

    X = pd.DataFrame(features)
    
    print(f"🤖 Training K-Means model with {n_clusters} clusters on {len(X)} samples...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Attach the assigned cluster ID back to each original job document
    clustered_jobs = []
    for job, label in zip(enriched_jobs, cluster_labels):
        job["cluster_id"] = int(label)
        clustered_jobs.append(job)

    # Save final dataset to MongoDB
    db[target_collection].drop()
    if clustered_jobs:
        db[target_collection].insert_many(clustered_jobs)
        print(f"✅ Successfully clustered and saved {len(clustered_jobs)} jobs into '{target_collection}'.")
