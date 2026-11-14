import json
from datetime import datetime

from pydantic import BaseModel, Field
from typing import Optional

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.models import Variable
from google.cloud import storage

from scripts.utils_raw_data import KEYWORDS, fetch_adzuna, fetch_jobsuche
from scripts.cleaning import clean_job_records_and_save_local
from scripts.nlp_enrichment_task import run_nlp_enrichment_local
from scripts.ml_preprocessing import run_feature_engineering_local
from scripts.clustering import run_kmeans_clustering_and_upsert_mongo

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
        "sources": Param("1", description="1: Adzuna + AfA | 2: Only AfA")
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
        from pymongo import MongoClient
        from datetime import datetime

        source_choice = kwargs['params']['sources']
        unified_jobs = []
        adzuna_jobs = []
        jobsuche_jobs = []

        # 1. Fetch data
        if str(source_choice) == '1':
            print("Sourcing from both AfA and Adzuna.")
            jobsuche_jobs = fetch_jobsuche(KEYWORDS, max_jobs_per_keyword=1000)
            adzuna_jobs = fetch_adzuna(KEYWORDS, max_jobs_per_keyword=1000)
            unified_jobs.extend(jobsuche_jobs)
            unified_jobs.extend(adzuna_jobs)
        else:
            print("Sourcing from AfA only.")
            jobsuche_jobs = fetch_jobsuche(KEYWORDS, max_jobs_per_keyword=1000)
            unified_jobs.extend(jobsuche_jobs)

        # 2. Count the jobs
        adzuna_counts = dict(Counter([job.get("job_title", "Unknown") for job in adzuna_jobs])) if adzuna_jobs else {}
        jobsuche_counts = dict(Counter([job.get("job_title", "Unknown") for job in jobsuche_jobs])) if jobsuche_jobs else {}

        # 3. Save the statistics to MongoDB
        try:
            # Update connection string if your auth/host differs
            mongo_client = MongoClient("mongodb://admin:admin@local_mongo_db:27017")
            db = mongo_client["job_market_db"]

            stats_doc = {
                "run_date": datetime.now(),
                "total_ingested": len(unified_jobs),
                "adzuna_breakdown": adzuna_counts,
                "jobsuche_breakdown": jobsuche_counts
            }

            db["ingestion_stats"].insert_one(stats_doc)
            print("Successfully logged ingestion stats to MongoDB.")
        except Exception as e:
            print(f"Failed to log stats to MongoDB: {e}")

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
        """Applies K-Means clustering and performs final UPSERT to local MongoDB."""
        run_kmeans_clustering_and_upsert_mongo(features_file_path, enriched_file_path, n_clusters=5)

    # Define full DAG task dependencies and pass file paths via XComs
    raw_file = ingest_jobs()
    cleaned_file = clean_data(raw_file)
    enriched_file = nlp_enrichment(cleaned_file)
    features_file = ml_feature_engineering(enriched_file)
    ml_clustering_and_load(features_file, enriched_file)

job_market_pipeline()

# 1. Define the value object structure
class SalaryMetrics(BaseModel):
    median: Optional[float] = None
    p25: Optional[float] = None  # 25. Percentile
    p75: Optional[float] = None  # 75. Percentile
    raw_error: Optional[str] = None

class BatchRequest(BaseModel):
    # User only provides the list of 5-digit KldB codes
    job_codes: List[str]

# Basis-URL der Bundesagentur für Arbeit
BA_BASE_URL = "https://arbeitsagentur.de"

async def fetch_single_job(
    client: httpx.AsyncClient,
    job_code: str,
    params: dict,
    semaphore: asyncio.Semaphore #prevents sending too many requests at once
) -> tuple[str, SalaryMetrics]:
    #Fetches salary data for one job and parses the median values.
    url = f"{BA_BASE_URL}/{job_code}"
    key = f"{name} ({job_code})"
    headers = {
            "X-API-Key": "infosysbub-ega",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }
    async with semaphore:
        try:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            # The BA API typically returns a list of regional/demographic entries.
            # Usually the first item matches the requested filter combination.
            entry = data[0] if isinstance(data, list) and len(data) > 0 else data

            # Extract the official neutral short title from the response
            job_name = entry.get("kurzBezeichnungNeutral") or "Unknown"
            key = f"{job_name} ({job_code})"

            metrics = SalaryMetrics(
                median=entry.get("entgelt"),
                lower_quartile=entry.get("entgeltQ25"),
                upper_quartile=entry.get("entgeltQ75"),
            )
            return key, metrics

        except Exception as exc:
            # Fallback key if the API call fails before kurzBezeichnungNeutral can be read
            fallback_key = f"Error ({job_code})"
            return fallback_key, SalaryMetrics(raw_error=str(exc))
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Fehler von der BA-API: {exc.response.text}",
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Verbindungsfehler zur BA-API: {str(exc)}",
            )

#"Data Engineer (43214)", "Data Scientist (43104)", "Software Engineer (43414)", "Machine Learning Engineer (134292)",
#"Data Analyst (138079)", "Cloud Engineer (133473 called Cloud-Architect @BA) ", "DevOps Engineer (133692)"
#"Backend Developer (14051 @BA Anwendungsprogrammierer/in)", "Business Intelligence Developer(Not found @BA)", "Analytics Engineer(Not found @BA)"

@dag(
    dag_id="german_median_salary_pipeline",
    schedule_interval="@daily",
    start_date=datetime(2023, 10, 1),
    catchup=False,
    params={
        "job_codes": Param("43214, 43104, 43414, 134292, 138079, 133473, 133692, 14051", description="Entgeltatlas specific Job Codes")
        "region": Param("1", description="1 = Germany, 2-17 = State")
        "gender": Param("1", description="1 = All, 2 = Men, 3 = Women")
        "age": Param("1", description="1 = All, 2 = under 25, 3 = 25-55, 4 = 56+")
        "branch": Param("1", description="1 = All")
    },
    tags=["job_market", "salary"]
)
def salary_pipeline():

    @task
    def retrieval(**kwargs):

        """Fetches salary data from Entgeltatlas, saves data to MongoDB"""
        from pymongo import MongoClient
        from datetime import datetime

        job_codes = kwargs['params']['job_codes'].replace(' ', '').split(',')

        params = {
            "r": kwargs['params']['region'],
            "g": kwargs['params']['gender'],
            "a": kwargs['params']['age'],
            "b": kwargs['params']['branch'],
        }    

        # 1. Fetch data

        semaphore = asyncio.Semaphore(5)
        results = {}
        salary_dict = {}

        async with httpx.AsyncClient(timeout=15.0) as client:
            tasks = [
                fetch_single_job(client, code, params, semaphore)
                for code in job_codes
            ]
            # Run all HTTP requests concurrently
            results = await asyncio.gather(*tasks)
            
        salary_dict: Dict[str, SalaryMetrics] = dict(results)

        # 2. Save the salary statistics to MongoDB
        try:
            # Update connection string if your auth/host differs
            mongo_uri = os.environ.get("MONGO_URI")
            if not mongo_uri:
                    return print('No Mongo URI found')
            mongo_client = MongoClient(mongo_uri)
            db = mongo_client["job_market_db"]

            stats_doc = {
                "run_date": datetime.now(),
                "salaryData": salary_dict
            }

            db["salary_stats"].insert_one(stats_doc)
            print("Successfully logged salary stats to MongoDB.")
        except Exception as e:
            print(f"Failed to log stats to MongoDB: {e}")

    @task
    def clean_data(raw_file_name: str):
        """Downloads GCS file, cleans it, and saves to LOCAL Docker volume."""
        client = storage.Client.from_service_account_json('/opt/airflow/dags/gcp-key.json', project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(raw_file_name)
        raw_jobs = json.loads(blob.download_as_string())

        cleaned_file_path = clean_job_records_and_save_local(raw_jobs, "/app/clean_data/cleaned_temp.parquet")
        return cleaned_file_path

    # Define full DAG task dependencies and pass file paths via XComs
    raw_file = ingest_jobs()

salary_pipeline()
