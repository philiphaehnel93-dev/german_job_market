# Job Market Data Pipeline & Dashboard

The primary goal of this data engineering pipeline is to automatically ingest, secure, and transform messy, semi-structured job market data into a highly structured format. By moving data from public APIs through an immutable Data Lake and a NoSQL staging environment, we can apply robust Natural Language Processing (NLP) to extract precise technical skills. This rigorous engineering process is required to turn raw text into clean, tabular, and vectorized features ready for machine learning (topic modeling) and advanced analytics.

[Click to see tech stack](#core-tech-stack)

*Note: `secrets.txt` and `.env` files are provided upon request and not available from the repository.*

## Quick Start / Setup Instructions

If you are a collaborator setting up this project locally, please follow these steps to spin up the Airflow pipeline and the Streamlit dashboard.

1. **Add GCP Credentials:** Drop your `gcp-key.json` file directly into the `airflow_project/dags/` folder.

2. **Configure Database Access:** Whitelist your local IP address in the MongoDB Atlas console (under Network Access) so your containers can connect to the database.

3. **Launch the Airflow Pipeline:**
   Navigate to the orchestration directory and initialize the Docker containers:

   ```
   cd airflow_project
   docker compose up airflow-init
   docker compose up -d
   ```
   In the Airflow console upload to Admin/Variables the variable.json that is to be replace with a different storage solution.

4. **Build the Dashboard Image:**
   Open a new terminal, navigate to the visualization folder, and build the Docker image:

   ```
   cd dashboard
   docker build -t job-dashboard .
   ```

5. **Run the Dashboard:**
   Start the visualization container, injecting the MongoDB connection string:

   ```
   docker run -d -p 8501:8501 -e MONGO_URI="<insert_uri_here>" --name my-dashboard job-dashboard
   ```
 ## Accessing the Dashboard (SSH Tunnel Required)

 Due to AWS Security Group firewall restrictions, direct web access to the Streamlit dashboard is blocked. To view the dashboard locally, you must create a secure SSH tunnel to the EC2 instance.

 ### Prerequisites
 - The AWS SSH key (`.pem` file) with restricted permissions (e.g., `chmod 400 <key_name>.pem` on Mac/Linux).
 - The public IP address of the EC2 instance.

 ### Steps to Connect

 **Create the tunnel:** Open a new terminal window on your local machine (do not run this on the server) and execute the following command, replacing the placeholders with your actual key path and server IP:
   ```bash
   ssh -i <path_to_your_key.pem> -L 8501:localhost:8501 ubuntu@<ec2_public_ip>
   ```

   *You can now view the dashboard locally at `http://localhost:8501`.*

## Pipeline Architecture & Execution Order

The project is divided into two main components: `airflow_project/` (Orchestration) and `dashboard/` (Visualization). The ETL and ML workflows are entirely automated by Apache Airflow.

### 1. Orchestration (`airflow_project/dags/`)

* **`job_market_dag.py`**: The main Airflow DAG (Directed Acyclic Graph) that defines the execution order, scheduling, and dependencies of all the pipeline tasks. Once Airflow is running, this DAG automatically orchestrates the steps below.

### 2. Task Scripts (`airflow_project/dags/scripts/`)

Airflow executes the following modular scripts in sequence:

* **Data Ingestion & Loading:**
  * **`utils_raw_data.py`**: Handles API requests, data gathering, and interacting with Google Cloud Storage.
  * **`mongo_loader.py`**: Connects to MongoDB Atlas to stage the raw, unstructured JSON payloads.

* **Cleaning & NLP Enrichment:**
  * **`cleaning.py`**: Prepares and sanitizes the raw text data before passing it to the NLP engine.
  * **`tech_dictionary.json`**: The static taxonomy of target technologies used for matching.
  * **`extractor.py`**: The core NLP engine utilizing FlashText and spaCy.
  * **`nlp_enrichment_task.py`**: The Airflow task wrapper that applies the extractor to the cleaned MongoDB documents, saving the results as structured entities.

* **Feature Engineering & Machine Learning:**
  * **`ml_preprocessing.py`**: Normalizes nested JSON into a flat structure, creates new variables, and applies vectorization (e.g., MultiLabelBinarizer for skills).
  * **`clustering.py`**: Fits unsupervised ML models (like K-Means) to the vectorized data to automatically discover job archetypes. It evaluates the distinctness of clusters (Silhouette Score) and saves the final dataset to MongoDB.

### 3. Visualization (`dashboard/`)

* **`app.py`**: A standalone Streamlit application that reads the final, clustered data directly from MongoDB Atlas to display market segments, top dynamic skills per cluster, and statutory salary baselines.

## Core Tech Stack

* **Docker & Apache Airflow:** Used for containerization, environment isolation, and orchestrating the execution dependencies of the ETL pipeline.

* **Streamlit:** The frontend Python framework used to build the interactive market segmentation dashboard.

* **Python 3.10+ & uv:** The foundation of the pipeline, utilizing `uv` as a blazingly fast package manager.

* **Google Cloud Storage (GCP):** Acts as the immutable Cloud Data Lake, providing a secure, historical backup of all raw API payloads.

* **MongoDB Atlas:** A fully managed NoSQL document database. It serves as the operational staging layer for raw data and the serving layer for the final dashboard.

* **spaCy & FlashText:** The hybrid NLP engine. FlashText provides lightning-fast exact string matching, while spaCy handles complex grammatical context.

* **Pandas & Scikit-Learn:** The feature engineering backbone used to flatten nested NoSQL documents, apply vectorization, and perform unsupervised ML clustering.
