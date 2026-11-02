import os
import time
import concurrent.futures
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Import your hybrid extractor
from scripts.extractor import HybridJobExtractor

# Global worker for multiprocessing
worker_extractor = None

def init_worker():
    """Initializes the NLP models once per process to avoid memory overhead."""
    global worker_extractor
    # Path relative to Airflow root/dags folder
    dict_path = os.path.join(os.path.dirname(__file__), "tech_dictionary.json")
    worker_extractor = HybridJobExtractor(dict_path=dict_path)

def process_single_job(job_doc: dict) -> dict:
    """Extracts entities from the job description and merges them."""
    description = job_doc.get("description", "")
    extracted_data = worker_extractor.process_job_description(description)
    
    job_doc["tech_stack"] = extracted_data.tech_stack
    job_doc["emails"] = extracted_data.emails
    job_doc["location_data"] = {
        "remote": extracted_data.location.get("remote", False),
        "city": extracted_data.location.get("city", ""),
        "lat": extracted_data.location.get("lat", None),
        "lon": extracted_data.location.get("lon", None)
    }
    return job_doc

def run_nlp_enrichment_pipeline():
    """Airflow-compatible wrapper for MongoDB NLP enrichment."""
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise ValueError("Error: MONGO_URI environment variable not set.")

    db_name = "job_market_db"
    raw_collection_name = "jobs_cleaned"  # Pulling from our Silver Layer!
    enriched_collection_name = "jobs_enriched"

    print(f"🚀 Connecting to MongoDB...")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
    except ConnectionFailure as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    db = client[db_name]
    
    print(f"📥 Fetching records from '{raw_collection_name}'...")
    raw_jobs = list(db[raw_collection_name].find({"description": {"$exists": True, "$ne": ""}}, {"_id": 0}))
    
    if not raw_jobs:
        print("⚠️ No job descriptions found to process.")
        return

    print(f"🚀 Starting parallel extraction for {len(raw_jobs)} jobs...")
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:
        enriched_jobs = list(executor.map(process_single_job, raw_jobs, chunksize=10))
        
    end_time = time.time()
    print(f"✅ Extraction finished in {end_time - start_time:.2f} seconds.")
    
    print(f"☁️ Uploading enriched data to MongoDB collection '{enriched_collection_name}'...")
    db[enriched_collection_name].drop() # Clear old runs
    if enriched_jobs:
        db[enriched_collection_name].insert_many(enriched_jobs)
        print(f"✅ Successfully loaded {len(enriched_jobs)} enriched records.")
