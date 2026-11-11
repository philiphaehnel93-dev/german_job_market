import os
import time
import concurrent.futures
import pandas as pd

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

def run_nlp_enrichment_local(input_path: str, output_path: str) -> str:
    """Reads cleaned data from local Parquet, extracts entities, and saves enriched file locally."""
    print(f"📥 Loading cleaned records from {input_path}...")
    df = pd.read_parquet(input_path)
    raw_jobs = df.to_dict(orient="records")

    if not raw_jobs:
        print("⚠️ No job descriptions found to process.")
        # Create an empty parquet file to prevent downstream task failures
        directory = os.path.dirname(output_path)
        os.makedirs(directory, exist_ok=True)
        pd.DataFrame().to_parquet(output_path, index=False)
        return output_path

    print(f"🚀 Starting parallel extraction for {len(raw_jobs)} jobs...")
    start_time = time.time()

    # Initialize the NLP models
    init_worker()

    # Execute extraction in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        enriched_jobs = list(executor.map(process_single_job, raw_jobs, chunksize=10))

    end_time = time.time()
    print(f"✅ Extraction finished in {end_time - start_time:.2f} seconds.")

    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    print(f"☁️ Saving enriched data to {output_path}...")
    enriched_df = pd.DataFrame(enriched_jobs)
    enriched_df.to_parquet(output_path, index=False)
    print(f"✅ Successfully saved {len(enriched_jobs)} enriched records.")

    return output_path
