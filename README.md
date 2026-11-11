# Job Market Data Pipeline & Dashboard

The primary goal of this data engineering pipeline is to automatically ingest, secure, and transform messy, semi-structured job market data into a highly structured format. By moving data from public APIs through an immutable Cloud Data Lake and generating intermediate `.parquet` files, we apply robust Natural Language Processing (NLP) to extract precise technical skills. This engineering process turns raw text into clean, tabular, and vectorized features ready for machine learning (topic modeling) and advanced analytics.

## Quick Start / Setup Instructions

If you are a collaborator setting up this project locally, please follow these steps to spin up the Airflow pipeline and the local database.

1. **Add GCP Credentials:** Drop your `gcp-key.json` file directly into the `airflow_project/dags/` folder. This is required for both the raw Data Lake and the ML Model Registry.

2. **Launch the Infrastructure:**
   Navigate to the root directory and start all microservices (Airflow, Local MongoDB, and local volumes). Initialization is handled automatically.

   ```bash
   docker-compose up -d
   ```

3. **Configure Airflow Variables:**
   Navigate to the Airflow UI (`http://localhost:8080`), go to Admin -> Variables, and upload your `variables.json` file to configure the storage paths and API keys.

4. **Run the Dashboard (Optional/Standalone):**
   If you want to run the Streamlit dashboard locally against your Docker database:

   ```bash
   cd dashboard
   docker build -t job-dashboard .
   docker run -d -p 8501:8501 -e MONGO_URI="mongodb://admin:admin@host.docker.internal:27017/" --name my-dashboard job-dashboard
   ```

## Accessing the Dashboard (EC2 Production Only)

Due to AWS Security Group firewall restrictions, direct web access to the Streamlit dashboard on a production VM is blocked. To view the dashboard locally from a remote server, create a secure SSH tunnel.

**Create the tunnel:** Open a new terminal window on your local machine and execute the following command, replacing the placeholders with your actual key path and server IP:
```bash
ssh -i <path_to_your_key.pem> -L 8501:localhost:8501 ubuntu@<ec2_public_ip>
```
*You can now view the dashboard locally at `http://localhost:8501`.*

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
  * **`clustering.py`**: Connects to the GCS Model Registry. It downloads the static `kmeans_production_model.joblib` (or trains a new one if missing) to predict job archetypes. It evaluates the Silhouette Score, then bulk-upserts the final clustered dataset and the ML performance metrics to local MongoDB.

### 3. Visualization (`dashboard/` & API)

* **FastAPI (Upcoming):** A microservice layer designed to read the `ml_metrics_history` collection from MongoDB and serve the historical model performance to the frontend.
* **`app.py`**: A Streamlit application that reads the final, clustered data directly from the database to display market segments, top dynamic skills, and salary baselines.

## Core Tech Stack

* **Docker & Apache Airflow:** Used for containerization, automated environment initialization, and orchestrating ETL dependencies.
* **Python 3.10+ & uv:** The foundation of the pipeline, utilizing `uv` for high-performance dependency management.
* **Google Cloud Storage (GCP):** Acts as the immutable Data Lake for raw JSONs and the centralized Model Registry for `.joblib` files.
* **MongoDB (Local Docker):** A NoSQL document database serving as the serving layer for the dashboard and ML metrics tracking.
* **spaCy & FlashText:** The hybrid NLP engine for lightning-fast exact string matching and grammatical context parsing.
* **Pandas & Scikit-Learn:** The feature engineering backbone for flattening nested documents, generating Parquet files, and performing K-Means clustering.
* **Streamlit:** The frontend Python framework used to build the interactive market segmentation dashboard.
