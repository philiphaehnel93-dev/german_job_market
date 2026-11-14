import os
import time
import concurrent.futures
import pandas as pd
import requests
from airflow.operators.python import get_current_context

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
    """Reads cleaned data, drops existing jobs, extracts entities, and saves locally."""
    print(f"📥 Loading cleaned records from {input_path}...")
    df = pd.read_parquet(input_path)

    if df.empty:
        print("⚠️ No job descriptions found to process.")
        pd.DataFrame().to_parquet(output_path, index=False)
        return output_path

    # --- INCREMENTAL FILTERING LOGIC ---
    print("🔍 Fetching existing job IDs from FastAPI to prevent reprocessing...")
    try:
        # Use the API_SECRET_KEY to authenticate the internal request
        token = os.getenv("API_SECRET_KEY")
        headers = {"Authorization": f"Bearer {token}"}

        response = requests.get(
            "http://airflow-fastapi:8081/api/v1/jobs/existing_ids",
            headers=headers,
            timeout=15
        )
        response.raise_for_status()
        existing_ids = set(response.json())

        # Filter the dataframe: Keep only rows where origin_id is NOT in existing_ids
        df_new = df[~df['origin_id'].isin(existing_ids)]
        print(f"📉 Filtered out {len(df) - len(df_new)} existing jobs.")

        raw_jobs = df_new.to_dict(orient="records")
    except Exception as e:
        print(f"⚠️ Failed to fetch existing IDs, defaulting to full reprocessing: {e}")
        raw_jobs = df.to_dict(orient="records")
    # -----------------------------------

    if not raw_jobs:
        print("✅ No new jobs to process today. Exiting early.")
        pd.DataFrame().to_parquet(output_path, index=False)

        # (Optional) Log pipeline metrics for 0 rows processed here if desired
        return output_path

    print(f"🚀 Starting extraction for {len(raw_jobs)} NEW jobs...")
    start_time = time.time()

    # Initialize the NLP models
    init_worker()

    # Execute extraction in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        enriched_jobs = list(executor.map(process_single_job, raw_jobs, chunksize=10))

    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"✅ Extraction finished in {duration} seconds.")

    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    print(f"☁️ Saving enriched data to {output_path}...")
    enriched_df = pd.DataFrame(enriched_jobs)
    enriched_df.to_parquet(output_path, index=False)
    print(f"✅ Successfully saved {len(enriched_jobs)} enriched records.")

    # Get Airflow context to log metrics to FastAPI endpoint
    context = get_current_context()
    run_id = context['dag_run'].run_id
    task_id = context['task'].task_id

    print(f"📡 Sending pipeline metrics to FastAPI for task: {task_id}...")
    try:
        response = requests.post(
            "http://airflow-fastapi:8081/api/v1/pipeline_metrics",
            json={
                "dag_id": "german_job_market_pipeline",
                "run_id": run_id,
                "task_id": task_id,
                "status": "success",
                "rows_processed": len(enriched_jobs),
                "duration_seconds": duration,
                "execution_date": str(pd.Timestamp.utcnow())
            },
            timeout=5
        )

        # Raise an exception if FastAPI returns a 4xx or 5xx error code
        response.raise_for_status()
        print(f"✅ Pipeline metrics successfully logged! (Status: {response.status_code})")

    except Exception as e:
        print(f"❌ Failed to log pipeline metrics: {e}")
        raise

    return output_path
