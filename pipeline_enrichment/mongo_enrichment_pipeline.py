import os
import sys
import json
import time
import concurrent.futures
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Ensure the extractor module is accessible
from extractor import HybridJobExtractor

# Load environment variables
load_dotenv()

# Configuration
DB_NAME = "job_market_db"
RAW_COLLECTION = "jobs_raw"
ENRICHED_COLLECTION = "jobs_enriched"
LOCAL_OUTPUT_FILE = "data_enriched.json"

# Global worker for multiprocessing
worker_extractor = None

def init_worker():
    """Initializes the NLP models once per process to avoid memory overhead and pickling errors."""
    global worker_extractor
    # Loads the strict tech_dictionary.json mappings and spaCy models into the independent memory space of each CPU core
    worker_extractor = HybridJobExtractor(dict_path="tech_dictionary.json")

def process_single_job(job_doc: dict) -> dict:
    """Extracts entities from the job description and merges them with the original document."""
    description = job_doc.get("description", "")
    
    # Process text using FlashText for unambiguous terms and spaCy for contextual terms
    extracted_data = worker_extractor.process_job_description(description)
    
    # Update the document with extracted entities
    job_doc["tech_stack"] = extracted_data.tech_stack
    job_doc["emails"] = extracted_data.emails
    job_doc["location_data"] = {
        "remote": extracted_data.location.get("remote", False),
        "city": extracted_data.location.get("city", ""),
        "lat": extracted_data.location.get("lat", None),
        "lon": extracted_data.location.get("lon", None)
    }
    return job_doc

def get_mongo_client() -> MongoClient:
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        print("Error: MONGO_URI environment variable not set.")
        sys.exit(1)
        
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client
    except ConnectionFailure:
        print("Error: Failed to connect to MongoDB Atlas.")
        sys.exit(1)

if __name__ == '__main__':
    client = get_mongo_client()
    db = client[DB_NAME]
    
    print(f"📥 Fetching records from '{RAW_COLLECTION}'...")
    raw_jobs = list(db[RAW_COLLECTION].find({"description": {"$exists": True, "$ne": ""}}, {"_id": 0}))
    
    if not raw_jobs:
        print("⚠️ No job descriptions found to process.")
        sys.exit(1)
        
    print(f"🚀 Starting parallel extraction for {len(raw_jobs)} jobs...")
    start_time = time.time()
    
    # Spawns separate Python processes to scale execution and saturate hardware
    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:
        # Chunks the data across available cores
        enriched_jobs = list(executor.map(process_single_job, raw_jobs, chunksize=10))
        
    end_time = time.time()
    print(f"✅ Extraction finished in {end_time - start_time:.2f} seconds.")
    
    # Save locally to JSON as requested
    with open(LOCAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched_jobs, f, indent=4, ensure_ascii=False)
    print(f"💾 Saved enriched data locally to '{LOCAL_OUTPUT_FILE}'")
    
    # Upload back to MongoDB Atlas
    print(f"☁️ Uploading enriched data to MongoDB collection '{ENRICHED_COLLECTION}'...")
    db[ENRICHED_COLLECTION].drop() # Clear old runs
    if enriched_jobs:
        db[ENRICHED_COLLECTION].insert_many(enriched_jobs)
    
    print("🎉 Pipeline complete!")
