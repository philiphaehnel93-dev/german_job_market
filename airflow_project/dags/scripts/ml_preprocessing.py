import os
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def run_feature_engineering():
    """Flattens enriched job records and multi-hot encodes tech stack for ML clustering."""
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise ValueError("Error: MONGO_URI environment variable not set.")

    db_name = "job_market_db"
    source_collection = "jobs_enriched"
    target_collection = "jobs_features"

    print("🚀 Connecting to MongoDB for feature engineering...")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
    except ConnectionFailure as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    db = client[db_name]
    
    # Fetch enriched jobs
    jobs = list(db[source_collection].find({}, {"_id": 0}))
    if not jobs:
        print("⚠️ No enriched jobs found to process.")
        return

    df = pd.DataFrame(jobs)
    
    print("🧹 Flattening and encoding features...")
    
    # 1. Flatten location data fields safely (extracting 'remote' boolean as integer 0/1)
    df['is_remote'] = df['location_data'].apply(
        lambda x: int(x.get('remote', False)) if isinstance(x, dict) else 0
    )
    
    # 2. Multi-hot encode the 'tech_stack' list into binary dummy columns
    if 'tech_stack' in df.columns:
        # Explodes lists into rows, generates dummy columns, then groups back by original index and sums
        tech_dummies = df['tech_stack'].explode().str.get_dummies().groupby(level=0).sum()
    else:
        tech_dummies = pd.DataFrame(index=df.index)

    # 3. Combine binary and numerical features into a clean feature matrix
    feature_df = pd.concat([df[['is_remote']], tech_dummies], axis=1).fillna(0)
    
    # Convert dataframe rows back to dictionaries for MongoDB insertion
    feature_records = feature_df.to_dict(orient='records')
    
    # Save feature matrix rows to MongoDB
    db[target_collection].drop()
    if feature_records:
        db[target_collection].insert_many(feature_records)
        print(f"✅ Successfully engineered and saved {len(feature_records)} feature rows into '{target_collection}'.")
