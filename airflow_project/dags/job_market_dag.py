import json
from datetime import datetime

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.models import Variable

from scripts.utils_raw_data import KEYWORDS, fetch_adzuna, fetch_jobsuche
from scripts.cleaning import transfer_gcs_to_mongo
from scripts.nlp_enrichment_task import run_nlp_enrichment_pipeline

from scripts.ml_preprocessing import run_feature_engineering
from scripts.clustering import run_kmeans_clustering

# Cloud Configuration fetched from Airflow Variables
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
        """Fetches job data from specified APIs and uploads raw payload to GCS."""
        source_choice = kwargs['params']['sources']
        unified_jobs = []

        if str(source_choice) == '1':
            print("Sourcing from both AfA and Adzuna.")
            unified_jobs.extend(fetch_adzuna(KEYWORDS, max_total_jobs=2000))
        else:
            print("Sourcing only from AfA.")

        unified_jobs.extend(fetch_jobsuche(KEYWORDS, max_total_jobs=1000))
        print(f"Pipeline Complete. Ingested {len(unified_jobs)} total raw jobs.")

        if unified_jobs:
            from google.cloud import storage
            client = storage.Client(project=PROJECT_ID)
            bucket = client.bucket(BUCKET_NAME)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"raw_jobs_data_{timestamp}.json"
            blob = bucket.blob(file_name)

            json_data = json.dumps(unified_jobs, ensure_ascii=False, indent=2)
            blob.upload_from_string(json_data, content_type="application/json")
            print(f"Successfully uploaded to gs://{BUCKET_NAME}/{file_name}")
            return file_name

    @task
    def clean_and_load_mongo(raw_file_path: str):
        """Downloads GCS file, cleans it, and loads into MongoDB (jobs_cleaned)."""
        transfer_gcs_to_mongo(
            project_id=PROJECT_ID,
            bucket_name=BUCKET_NAME,
            file_name=raw_file_path
        )

    @task
    def nlp_enrichment():
        """Pulls from jobs_cleaned, extracts tech/locations via spaCy/FlashText, saves to jobs_enriched."""
        run_nlp_enrichment_pipeline()

    @task
    def ml_feature_engineering():
        """Prepares numerical feature matrices from enriched records."""
        run_feature_engineering()

    @task
    def ml_clustering():
        """Applies K-Means clustering and stores final tagged records."""
        run_kmeans_clustering(n_clusters=5)

    # Define full DAG task dependencies
    raw_file = ingest_jobs()
    loaded_mongo = clean_and_load_mongo(raw_file)
    enriched_mongo = nlp_enrichment()
    features_built = ml_feature_engineering()
    clustered_data = ml_clustering()

    # Execution chain sequence
    loaded_mongo >> enriched_mongo >> features_built >> clustered_data
