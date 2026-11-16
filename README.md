# Job Market Data Pipeline & Dashboard

The primary goal of this data engineering pipeline is to automatically ingest, secure, and transform messy, semi-structured job market data into a highly structured format. By moving data from public APIs through an immutable Cloud Data Lake and generating intermediate `.parquet` files, we apply robust Natural Language Processing (NLP) to extract precise technical skills. This engineering process turns raw text into clean, tabular, and vectorized features ready for machine learning (topic modeling) and advanced analytics.

## Quick Start / Setup Instructions

If you are a collaborator setting up this project locally, please follow these steps to spin up the Airflow pipeline and the local database.

1. **Add GCP Credentials:** Drop your `gcp-key.json` file directly into the `airflow_project/dags/` folder. This is required for both the raw Data Lake and the ML Model Registry.

2. **Launch the Infrastructure:**
   Navigate to the root directory and start all microservices (Airflow, FastAPI, Local MongoDB, and Streamlit). Initialization is handled automatically.

   ```bash
   docker-compose up -d
   ```

3. **Configure Environment Variables:**
Create a .env file in the root directory and configure the following variables:

External APIs & Cloud:
AIRFLOW_VAR_ADZUNA_APP_ID & AIRFLOW_VAR_ADZUNA_APP_KEY: Credentials for fetching job data from Adzuna.
AIRFLOW_VAR_GCP_PROJECT_ID, AIRFLOW_VAR_GCP_PROJECT_NUMBER, AIRFLOW_VAR_GCP_BUCKET_NAME: Google Cloud Storage configuration.
GOOGLE_APPLICATION_CREDENTIALS: Internal container path to the GCP key (e.g., /opt/airflow/dags/gcp-key.json).
Database (MongoDB):
MONGO_URI, MONGO_DB_NAME: Database connection string and targeted database name.
MONGO_INITDB_ROOT_USERNAME, MONGO_INITDB_ROOT_PASSWORD: Root credentials for database initialization.

FastAPI Security:
API_SECRET_KEY, FASTAPI_SECRET_KEY: Secret keys used for generating and validating JWTs for secure API access.

Airflow Initialization & Security:
AIRFLOW_USER, AIRFLOW_PASSWORD: Login credentials for the Airflow Web UI.
AIRFLOW__WEBSERVER__SECRET_KEY, AIRFLOW__CORE__FERNET_KEY: Cryptographic keys for securing the Airflow webserver and encrypting connections.
AIRFLOW__CORE__EXECUTOR, AIRFLOW__CORE__SQL_ALCHEMY_CONN: Airflow execution backend configuration.
AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION, AIRFLOW__CORE__LOAD_EXAMPLES, AIRFLOW__API__AUTH_BACKEND: General Airflow behavior settings.
_PIP_ADDITIONAL_REQUIREMENTS: Extra Python dependencies injected on startup (e.g., scikit-learn).

Airflow SMTP (Alerts):
AIRFLOW__SMTP__SMTP_HOST, AIRFLOW__SMTP__SMTP_PORT, AIRFLOW__SMTP__SMTP_USER, AIRFLOW__SMTP__SMTP_PASSWORD, AIRFLOW__SMTP__SMTP_MAIL_FROM: Email server settings for task failure notifications.
Pipeline Settings:
JOB_KEYWORD_LIMIT: Integer limit for NLP keyword extractions per job.

4. **Airflow:**
   The running Airflow pipeline is integrated into the `docker-compose` setup and is accessible from `http://localhost:8080`.

4. **Dashboard:**
   The running Streamlit dashboard is integrated into the `docker-compose` setup and is accessible from `http://localhost:8501`.

## Pipeline Architecture & Execution Order

The project is divided into orchestration, pipeline tasks, and visualization. The ETL and ML workflows are entirely automated by Apache Airflow.

### 1. Orchestration (`airflow_project/dags/`)

* **`job_market_dag.py`**: The main Airflow DAG defining the execution order, scheduling, and dependencies.

### 2. Task Scripts (`airflow_project/dags/scripts/`)

Airflow executes these modular scripts in sequence, passing data via local `.parquet` files mounted in Docker volumes.

* **Data Ingestion & Loading:**
  * **`utils_raw_data.py`**: Handles API requests and uploads raw JSON payloads to Google Cloud Storage (Data Lake).

* **Cleaning & NLP Enrichment:**
  * **`cleaning.py`**: Prepares and sanitizes the raw text data, creating the necessary local directories dynamically and saving to `cleaned_temp.parquet`.
  * **`extractor.py` / `tech_dictionary.json`**: The core NLP engine utilizing FlashText and spaCy against a static taxonomy.
  * **`nlp_enrichment_task.py`**: Applies the extractor and saves the structured entities to `enriched_temp.parquet`.

* **Feature Engineering & Machine Learning:**
  * **`ml_preprocessing.py`**: Normalizes nested data into a flat structure, applies vectorization (MultiLabelBinarizer), and saves to `features_temp.parquet`.
  * **`clustering.py`**: Connects to the GCS Model Registry. It downloads the static `kmeans_production_model.joblib`, uses a local model or trains a new one if it missing to predict job archetypes. It evaluates the Silhouette Score, then bulk-upserts the final clustered dataset and the ML performance metrics to FastAPI for ingestion into MongoDB.

### 3. Visualization (`dashboard/` & API)

* **FastAPI Middleware (`main.py`):** A fully functional, secure bridge between the Airflow pipeline and the backend database.
  * **Ingestion (Internal):** Handles bulk data upserts and metric logging from Airflow via unprotected internal Docker network routes.
  * **Retrieval (External/Secured):** Uses JWT Bearer authentication (`FASTAPI_SECRET_KEY`) to securely serve the clustered jobs and historical ML metrics to external consumers.
* **Streamlit Dashboard (`app.py`):** An interactive frontend application that fetches data securely from the FastAPI endpoints to display market segments, top dynamic skills, and salary baselines. It utilizes Streamlit's data caching mechanisms to optimize performance, with a built-in cache-clearing function to instantly pull live updates after a successful pipeline run.

## Core Tech Stack

* **Docker & Apache Airflow:** Used for containerization, automated environment initialization, and orchestrating ETL dependencies.
* **Python 3.10+ & uv:** The foundation of the pipeline, utilizing `uv` for high-performance dependency management.
* **FastAPI:** Acts as the high-performance middleware and API gateway, securing data retrieval with JWT authentication.
* **Google Cloud Storage (GCP):** Acts as the immutable Data Lake for raw JSONs and the centralized Model Registry for `.joblib` files.
* **MongoDB (Local Docker):** A NoSQL document database serving as the persistent storage layer for clustered jobs, ML metrics, and API access logs.
* **spaCy & FlashText:** The hybrid NLP engine for lightning-fast exact string matching and grammatical context parsing.
* **Pandas & Scikit-Learn:** The feature engineering backbone for flattening nested documents, generating Parquet files, and performing K-Means clustering.
* **Streamlit:** The frontend Python framework used to build the interactive market segmentation dashboard.

## .env

```AIRFLOW_VAR_GCP_PROJECT_ID=xx
AIRFLOW_VAR_GCP_PROJECT_NUMBER=xx
AIRFLOW_VAR_GCP_BUCKET_NAME=xx
AIRFLOW_VAR_ADZUNA_APP_ID=xx
AIRFLOW_VAR_ADZUNA_APP_KEY=xx

GOOGLE_APPLICATION_CREDENTIALS=xx

JOB_KEYWORD_LIMIT=10

MONGO_URI=xx
MONGO_INITDB_ROOT_USERNAME=xx
MONGO_INITDB_ROOT_PASSWORD=xx

AIRFLOW_USER=xx
AIRFLOW_PASSWORD=xx

MONGO_DB_NAME=xx

AIRFLOW__WEBSERVER__SECRET_KEY=xx
AIRFLOW__CORE__EXECUTOR=xx
AIRFLOW__CORE__SQL_ALCHEMY_CONN=xx
AIRFLOW__CORE__FERNET_KEY=""
AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=xx
AIRFLOW__CORE__LOAD_EXAMPLES=xx
AIRFLOW__API__AUTH_BACKEND=xx
AIRFLOW__SMTP__SMTP_HOST=xx
AIRFLOW__SMTP__SMTP_PORT=xx
AIRFLOW__SMTP__SMTP_USER=xx
AIRFLOW__SMTP__SMTP_PASSWORD=xx
AIRFLOW__SMTP__SMTP_MAIL_FROM=xx
_PIP_ADDITIONAL_REQUIREMENTS=${_PIP_ADDITIONAL_REQUIREMENTS:- scikit-learn}

API_SECRET_KEY=xx
FASTAPI_SECRET_KEY=xx
