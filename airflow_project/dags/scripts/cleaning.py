import os
import pandas as pd
import requests
from airflow.operators.python import get_current_context

def clean_job_records(raw_jobs: list) -> list:
    """Cleans, normalizes, and standardizes raw job dictionaries."""
    cleaned_jobs = []
    for job in raw_jobs:
        job_title = job.get("job_title", "Unspecified")
        if not job_title or job_title == "Unknown":
            continue  # Skip invalid entries

        cleaned_job = {
            "origin_id": job.get("origin_id"),
            "source_api": job.get("source_api", "Unknown"),
            "job_title": job_title.strip(),
            "company_name": job.get("company_name", "Unknown").strip(),
            "location": job.get("location", "Unknown").strip(),
            "is_remote": bool(job.get("is_remote", False)),
            "seniority_level": job.get("seniority_level", "Unspecified"),
            "description": job.get("description", "").strip(),
            "tech_stack": job.get("tech_stack", [])
        }
        cleaned_jobs.append(cleaned_job)

    return cleaned_jobs

def clean_job_records_and_save_local(raw_jobs: list, output_path: str) -> str:

    cleaned_jobs = clean_job_records(raw_jobs)

    # Save to local Docker volume instead of Mongo
    df = pd.DataFrame(cleaned_jobs)
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Get Airflow context to log metrics to FastAPI endpoint
    context = get_current_context()
    run_id = context['dag_run'].run_id
    task_id = context['task'].task_id

    try:
        response = requests.post(
            "http://airflow-fastapi:8081/api/v1/pipeline_metrics",
            json={
                "dag_id": "german_job_market_pipeline",
                "run_id": context['dag_run'].run_id,
                "task_id": context['task'].task_id,
                "status": "success",
                "rows_processed": len(cleaned_jobs),
                "duration_seconds": 0.0,
                "execution_date": str(pd.Timestamp.utcnow())
            },
            timeout=5
        )
        # Raise an exception if FastAPI returns a 4xx or 5xx error code
        response.raise_for_status()
        print(f"✅ Pipeline metrics successfully logged! (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Failed to log pipeline metrics: {e}")
        raise  # Forces the Airflow task to fail
    return output_path
