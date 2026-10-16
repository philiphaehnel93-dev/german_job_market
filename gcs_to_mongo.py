import json
import os
import sys
from google.cloud import storage
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Configuration
PROJECT_ID = "ml-pipeline-hu1tac"
BUCKET_NAME = f"{PROJECT_ID}-ml-data"
FILE_NAME = "raw_jobs_data.json"

DB_NAME = "job_market_db"
COLLECTION_NAME = "jobs_raw"

def move_gcs_to_mongo():
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        print("Error: MONGO_URI environment variable not set. Run export MONGO_URI=...")
        sys.exit(1)

    print(f"📥 Downloading '{FILE_NAME}' from gs://{BUCKET_NAME}...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(FILE_NAME)
    
    if not blob.exists():
        print(f"Error: Blob {FILE_NAME} does not exist in bucket {BUCKET_NAME}.")
        sys.exit(1)

    json_data = blob.download_as_string()
    jobs = json.loads(json_data)
    print(f"✅ Successfully loaded {len(jobs)} jobs into memory.")

    print(f"🚀 Connecting to MongoDB Atlas database '{DB_NAME}'...")
    try:
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')
    except ConnectionFailure:
        print("Error: Failed to connect to MongoDB Atlas. Check your MONGO_URI.")
        sys.exit(1)

    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]

    print("🧹 Clearing existing collection to prevent duplicates...")
    collection.delete_many({})

    if jobs:
        print("💾 Inserting jobs into MongoDB...")
        collection.insert_many(jobs)
        print(f"✅ Successfully inserted {len(jobs)} jobs into '{COLLECTION_NAME}'.")
    else:
        print("⚠️ No jobs found in the downloaded file.")

if __name__ == "__main__":
    move_gcs_to_mongo()
