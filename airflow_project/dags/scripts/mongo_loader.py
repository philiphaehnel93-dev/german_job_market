import os
from airflow.models import Variable
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def insert_cleaned_jobs_to_mongo(cleaned_jobs: list):
    """Connects to MongoDB Atlas and inserts cleaned job records."""
    mongo_uri = Variable.get("MONGO_URI")
    if not mongo_uri:
        raise ValueError("Error: MONGO_URI environment variable not set.")

    db_name = "job_market_db"
    collection_name = "jobs_cleaned"

    try:
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')
    except ConnectionFailure as e:
        raise ConnectionError(f"Failed to connect to MongoDB Atlas: {e}")

    db = mongo_client[db_name]
    collection = db[collection_name]

    print(f"🧹 Clearing existing '{collection_name}' collection...")
    collection.delete_many({})

    if cleaned_jobs:
        print(f"💾 Inserting {len(cleaned_jobs)} cleaned jobs into MongoDB...")
        collection.insert_many(cleaned_jobs)
        print("✅ Successfully loaded cleaned records.")
    else:
        print("⚠️ No valid jobs to insert.")
