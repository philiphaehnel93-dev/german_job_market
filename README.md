# Job Market ETL & Machine Learning Pipeline

A modular data engineering and machine learning pipeline designed to analyze the German tech job market. The system ingests job postings across public APIs, backs up raw data into a Google Cloud Storage (GCP) Data Lake, stages records in MongoDB Atlas, enriches documents using a multi-core hybrid NLP engine, and prepares vectorized features for unsupervised topic discovery.

For comprehensive architectural design, data flow lifecycle, and the microservices roadmap, please refer to **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Prerequisites

- **Python:** 3.10+
- **Package Manager:** `uv`
- **Cloud Accounts:** Google Cloud Platform (GCP) with an active billing account, MongoDB Atlas
- **System CLIs:** Google Cloud CLI (`gcloud`), MongoDB Atlas CLI (`atlas`), Homebrew (macOS)

---

## 1. Quickstart Setup Guide

### Step 1.1: Provision Google Cloud Infrastructure
Run the automated GCP provisioning script to create a project, link your billing account, and enable the Cloud Storage API:

```bash
bash setup_project.sh
```
Note: Save your new GCP Project ID output at the end of the script to use in your .env file.
Step 1.2: Configure MongoDB Atlas Cluster
Install the Atlas CLI and configure your cluster credentials:
# Install Atlas CLI (macOS)
brew install mongodb-atlas-cli

# Authenticate via API keys (Select "plaintext" format)
atlas setup

# Create a free M0 tier cluster (if you do not already have one)
atlas cluster create ml-pipeline-cluster --tier M0 --provider AWS --region EU_CENTRAL_1

# Create database user credentials
atlas dbusers create --username admin --password <YourSecurePassword> --role readWriteAnyDatabase

Step 1.3: Set Up Local Python Environment
Use uv to configure an isolated virtual environment and download the required spaCy German language model:
# Automated setup
bash setup_env.sh

# OR Manual setup via uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run python -m spacy download de_core_news_sm

Step 1.4: Configure Environment Variables
Create a .env file in the root of the project with your API keys, database connection string, and cloud credentials:
# MongoDB Atlas
MONGO_URI="mongodb+srv://admin:<YourSecurePassword>@<cluster-url>.mongodb.net/?retryWrites=true&w=majority"

# Adzuna API Credentials
ADZUNA_APP_ID="your_adzuna_app_id"
ADZUNA_APP_KEY="your_adzuna_app_key"

# Google Cloud
GCP_PROJECT_ID="your_gcp_project_id"
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/gcp-key.json"

2. Pipeline Execution Sequence
Execute the pipeline stages sequentially using uv run to ensure all processes execute safely within the managed virtual environment.

2.1. Ingestion to Cloud Data Lake
Provisions the standard GCS storage bucket and pulls job postings from the Adzuna and Bundesagentur für Arbeit (AfA) Jobsuche APIs:

uv run scripts/create_bucket.py
uv run scripts/ingest_jobs.py

2.2 Staging to Operational Database
Downloads raw payloads from the GCS bucket, clears existing staging collections to prevent duplicates, and inserts records into MongoDB Atlas:
uv run scripts/gcs_to_mongo.py

2.3 Multi-Core Hybrid NLP Enrichment
Runs parallel worker processes to extract emails, coordinates, unambiguous skills, and context-dependent programming languages. Outputs to the jobs_enriched collection and writes data_enriched.json:
uv run pipeline_enrichment/mongo_enrichment_pipeline.py

2.4 Exploratory Data Analysis (EDA)
Launch Jupyter to inspect job distributions, geographical densities, skill co-occurrences, and topic cluster boundaries:
uv run jupyter notebook job_market_eda.ipynb

TODO

2.5. Tabular Flattening & Machine Learning Preprocessing
Flattens polymorphic JSON structures into a tabular DataFrame, applies TF-IDF and multi-hot encoding, fits an NMF topic model, and exports data_vectorized.json:
uv run pipeline_enrichment/ml_preprocessing_pipeline.py

4. Next Steps & Architecture Roadmap
Web Scraping Integration (BeautifulSoup): Resolve Adzuna API snippet truncations by following redirect URLs to extract full HTML job posting bodies prior to raw lake ingestion.

Microservices Containerization (Docker): Package the pipeline stages (ingestion, staging, enrichment, ml_pipeline) into standalone, reproducible Docker containers to eliminate local dependency drift.

Pipeline Orchestration (Apache Airflow): Transition execution from manual CLI runs to scheduled Airflow DAGs, enabling automated daily triggering, task dependency enforcement, and built-in retries.

Serving Layer (FastAPI & Streamlit): Expose MongoDB collections and topic distribution endpoints via FastAPI, serving an interactive Streamlit UI for dynamic job market exploration.
