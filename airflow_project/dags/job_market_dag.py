import asyncio
import json
from datetime import datetime

import httpx
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.models.param import Param
from google.cloud import storage
from schemas import SalaryMetrics
from scripts.cleaning import clean_job_records_and_save_local
from scripts.clustering import run_kmeans_clustering_and_upsert_mongo
from scripts.ml_preprocessing import run_feature_engineering_local
from scripts.nlp_enrichment_task import run_nlp_enrichment_local
from scripts.utils_raw_data import KEYWORDS, fetch_adzuna, fetch_jobsuche
import requests

# Cloud Configuration
PROJECT_ID = Variable.get("GCP_PROJECT_ID")
PROJECT_NUMBER = Variable.get("GCP_PROJECT_NUMBER")
BUCKET_NAME = Variable.get("GCP_BUCKET_NAME")

@dag(
    dag_id="german_job_market_pipeline",
    schedule_interval="@daily",
    start_date=datetime(2023, 10, 1),
    catchup=False,
    params={
        "sources": Param("1", description="1: Adzuna + AfA | 2: Adzuna | 3: AfA ", type="string")
    },
    tags=["job_market", "ingestion"]
)
def job_market_pipeline():
    @task
    def ingest_jobs(**kwargs):
        print("\n====== CONFIGURATION CHECK ======")
        print(f"Total Keywords: {len(KEYWORDS)}")
        print(f"Keywords to Query: {KEYWORDS}")
        print("=================================\n")

        """Fetches job data, saves stats to MongoDB, and uploads payload to GCS."""
        from collections import Counter
        from datetime import datetime

        source_choice = kwargs['params']['sources']
        unified_jobs = []
        adzuna_jobs = []
        jobsuche_jobs = []
        # 1. Fetch data
        if str(source_choice) == '1':
            print("Sourcing from both AfA and Adzuna.")
            jobsuche_jobs = fetch_jobsuche(KEYWORDS, max_jobs_per_keyword=1000)
            adzuna_jobs = fetch_adzuna(KEYWORDS, max_jobs_per_keyword=2000)
            unified_jobs.extend(jobsuche_jobs)
            unified_jobs.extend(adzuna_jobs)
        elif str(source_choice) == '2':
            print("Sourcing from Adzuna only.")
            adzuna_jobs = fetch_adzuna(KEYWORDS, max_jobs_per_keyword=2000)
            unified_jobs.extend(adzuna_jobs)
        else:
            print("Sourcing from AfA only.")
            jobsuche_jobs = fetch_jobsuche(KEYWORDS, max_jobs_per_keyword=1000)
            unified_jobs.extend(jobsuche_jobs)
        # 2. Count the jobs
        adzuna_counts = dict(Counter([job.get("job_title", "Unknown") for job in adzuna_jobs])) if adzuna_jobs else {}
        jobsuche_counts = dict(Counter([job.get("job_title", "Unknown") for job in jobsuche_jobs])) if jobsuche_jobs else {}

        # 3. Save the statistics to MongoDB via FastAPI
        try:
            stats_payload = {
                "run_date": datetime.now().isoformat(),  # Converted to string for JSON serialization
                "total_ingested": len(unified_jobs),
                "adzuna_breakdown": adzuna_counts,
                "jobsuche_breakdown": jobsuche_counts
            }

            # Replace 'fastapi_container_name:8000' with your actual FastAPI service name and port in Docker
            api_url = "http://airflow-fastapi:8081/api/v1/ingestion_stats"

            response = requests.post(api_url, json=stats_payload)
            response.raise_for_status()  # Raises an exception for 4XX/5XX errors

            print("Successfully logged ingestion stats via FastAPI.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to log stats via FastAPI: {e}")
            raise

        # 4. Upload to GCS and return file name for the next task
        if unified_jobs:
            gcs_client = storage.Client.from_service_account_json('/opt/airflow/dags/gcp-key.json', project=PROJECT_ID)
            bucket = gcs_client.bucket(BUCKET_NAME)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"raw_jobs_data_{timestamp}.json"
            blob = bucket.blob(file_name)

            json_data = json.dumps(unified_jobs, ensure_ascii=False, indent=2)
            blob.upload_from_string(json_data, content_type="application/json")

            return file_name

    @task
    def clean_data(raw_file_name: str):
        """Downloads GCS file, cleans it, and saves to LOCAL Docker volume."""
        client = storage.Client.from_service_account_json('/opt/airflow/dags/gcp-key.json', project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(raw_file_name)
        raw_jobs = json.loads(blob.download_as_string())

        cleaned_file_path = clean_job_records_and_save_local(raw_jobs, "/app/clean_data/cleaned_temp.parquet")
        return cleaned_file_path

    @task
    def nlp_enrichment(cleaned_file_path: str):
        """Loads local cleaned data, extracts tech/locations, saves enriched file locally."""
        enriched_file_path = run_nlp_enrichment_local(cleaned_file_path, "/app/clean_data/enriched_temp.parquet")
        return enriched_file_path

    @task
    def ml_feature_engineering(enriched_file_path: str):
        """Prepares numerical feature matrices from local enriched records."""
        features_file_path = run_feature_engineering_local(enriched_file_path, "/app/clean_data/features_temp.parquet")
        return features_file_path

    @task
    def ml_clustering_and_load(features_file_path: str, enriched_file_path: str):
        """Applies K-Means clustering and performs final UPSERT to local MongoDB via FastAPI."""
        run_kmeans_clustering_and_upsert_mongo(features_file_path, enriched_file_path, n_clusters=5)

    # Define full DAG task dependencies and pass file paths via XComs
    raw_file = ingest_jobs()
    cleaned_file = clean_data(raw_file)
    enriched_file = nlp_enrichment(cleaned_file)
    features_file = ml_feature_engineering(enriched_file)
    ml_clustering_and_load(features_file, enriched_file)

job_market_pipeline()

# Basis-URL der Bundesagentur für Arbeit
BA_BASE_URL = "https://rest.arbeitsagentur.de/infosysbub/entgeltatlas/pc/v1/entgelte"

headers = {
        "X-API-Key": "infosysbub-ega", # The updated key
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-PL,en;q=0.9,de-DE;q=0.8,de;q=0.7,en-GB;q=0.6,en-US;q=0.5",
        "Origin": "https://web.arbeitsagentur.de",
        "Connection": "keep-alive",
        # Adding these modern browser headers heavily reduces bot detection
        "sec-ch-ua": '"Not-A.Brand";v="99", "Google Chrome";v="157", "Chromium";v="157"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"'
    }

async def fetch_single_job(
    url: str,
    client: httpx.AsyncClient,
    job_code: str,
    params: dict,
    semaphore: asyncio.Semaphore #prevents sending too many requests at once
) -> tuple[str, SalaryMetrics]:
    #Fetches salary data for one job and parses the median values.

    async with semaphore:
            url = f"{url}/{job_code}"
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            if response.status_code == 204 or not response.content.strip():
                print(
                    f"No salary data returned (HTTP {response.status_code}) for {response.request.url}"
                )
                return None

            try:
                data = response.json()
                return data
            except json.JSONDecodeError as exc:
                raise AirflowException(
                    f"Invalid JSON (HTTP {response.status_code}) from {response.request.url}. "
                    f"Body preview: {response.text[:200]!r}"
                ) from exc

async def fetch_all_jobs(url, params, job_codes):
    semaphore = asyncio.Semaphore(5)
    results = {}

    async with httpx.AsyncClient(timeout=15.0) as client:
        tasks = [
            fetch_single_job(url, client, code, params, semaphore)
            for code in job_codes
        ]
        # Run all HTTP requests concurrently
        results = await asyncio.gather(*tasks)

    return results

async def simple_async_post(url: str, payload: dict | None = None) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

#"Data Engineer (135872/43104)", "Data Scientist (129987/43104)", "Software Engineer (14051/43423 @BA Anwendungsprogrammierer/in)", "Machine Learning Engineer (134292/43104)",
#"Data Analyst (138079/43353)", "Cloud Engineer (133473/43414 called Cloud-Architect @BA) ", "DevOps Engineer (133692/43323)"
#"Backend Developer (135013/43413 als Backend-Entwickler/in)", "Business Intelligence Developer(Not found @BA)", "Analytics Engineer(Not found @BA)"

@dag(
    dag_id="german_median_salary_pipeline",
    schedule_interval="@daily",
    start_date=datetime(2023, 10, 1),
    catchup=False,
    params={
        "job_codes": Param("43104, 43423, 43353, 43414, 43323, 43413", description="Entgeltatlas specific Job Codes"),
        "region": Param("1", description="1 = Germany, 2-17 = State"),
        "gender": Param("1", description="1 = All, 2 = Men, 3 = Women"),
        "age": Param("1", description="1 = All, 2 = under 25, 3 = 25-55, 4 = 56+"),
        "branch": Param("1", description="1 = All")
    },
    tags=["job_market", "salary"]
)
def salary_pipeline():


    @task
    def execute_async_requests_to_entgeltatlas(**kwargs):

        """Fetches salary data from Entgeltatlas"""
        job_codes = kwargs['params']['job_codes'].replace(' ', '').split(',')
        params = {
            "r": kwargs['params']['region'],
            "g": kwargs['params']['gender'],
            "a": kwargs['params']['age'],
            "b": kwargs['params']['branch'],
        }    

        # 1. Fetch data
        url = BA_BASE_URL
        results = asyncio.run(fetch_all_jobs(url, params, job_codes))
        salary_dictionary = {}
        for result in results:
            entry = result[0] if isinstance(result, list) and len(result) > 0 else result
            key = entry.get("kldb")

            metrics = SalaryMetrics(
                median=entry.get("entgelt"),
                p25=entry.get("entgeltQ25"),
                p75=entry.get("entgeltQ75"),
            ).model_dump(mode="json")
            salary_dictionary[key] = metrics
            
        return salary_dictionary

    @task
    def save_salary_data(results: dict):

        dt = datetime.now()
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")

        salary_doc = {
                "fetch_date": dt_str,
                "salaryData": results
        }
        
        # 2. Save the salary statistics to MongoDB
        try:
            url = "http://airflow-fastapi:8081/api/v1/salary"
            asyncio.run(simple_async_post(url, salary_doc))
            print("Successfully logged salary stats to MongoDB.")
        except Exception as e:
            raise AirflowException(f"Failed to save salary stats to MongoDB: {e}")

    salary_data = execute_async_requests_to_entgeltatlas()
    save_salary_data(salary_data)

salary_pipeline()
