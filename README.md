# Pipeline purpose

The primary goal of this data engineering pipeline is to automatically ingest, secure, and transform messy, semi-structured job market data into a highly structured format. By moving data from public APIs through an immutable Data Lake and a NoSQL staging environment, we can apply robust Natural Language Processing (NLP) to extract precise technical skills. This rigorous engineering process is required to turn raw text into clean, tabular, and vectorized features ready for machine learning (topic modeling) and advanced analytics.

[Click to see tech stack](#core-tech-stack)

secrets.txt provided upon request and not avalable from repository

# Pipeline Script Execution Order

This document outlines the required execution sequence for the Job Market ETL pipeline. It is structured to support initial data exploration, text enrichment, and advanced feature engineering.

## Phase 1: Bootstrapping & Environment Setup

1. **`setup_project.sh`**
   * **Purpose:** Automates the creation of your Google Cloud Platform (GCP) project, links your billing account, and enables the Cloud Storage API.
   * **Output:** Generates the GCP Project ID required for your `.env` file.

2. **`setup_env.sh`**
   * **Purpose:** Initializes the local Python environment using the `uv` package manager. Installs dependencies and downloads the `de_core_news_sm` spaCy model.

## Phase 2: Ingestion & Cloud Storage (Root Directory)

3. **`create_bucket.py`**
   * **Purpose:** Authenticates with GCP and provisions the Data Lake storage bucket.

4. **`ingest_jobs.py`**
   * **Purpose:** Queries APIs, aggregates the raw JSON payload, and uploads it to the GCP bucket.

5. **`gcs_to_mongo.py`**
   * **Purpose:** Downloads raw data from GCP, clears existing staging collections, and inserts the unformatted data into MongoDB Atlas (`jobs_raw`).

## Phase 3: Initial EDA (Pre-NLP)

*Note: Before writing extraction rules, it is highly recommended to explore the `jobs_raw` collection to understand data quality, missing fields, and text distributions.*

6. **`job_market_eda.ipynb` (Part 1)**
   * **Purpose:** Connect to MongoDB's raw collection to perform preliminary EDA. Analyze job title frequencies, text lengths, and identify API limitations before applying NLP.

## Phase 4: Hybrid NLP Enrichment (`pipeline_enrichment/`)

7. **`pipeline_enrichment/tech_dictionary.json`** *(Dependency)*
   * **Purpose:** Static JSON containing the taxonomy of target technologies.

8. **`pipeline_enrichment/extractor.py`** *(Dependency)*
   * **Purpose:** The core NLP engine using FlashText and spaCy.

9. **`pipeline_enrichment/mongo_enrichment_pipeline.py`**
   * **Purpose:** Pulls raw documents from MongoDB, applies text extraction across multiple CPU cores, and saves structured entities to the `jobs_enriched` collection.

## Phase 5: Flattening & Feature Engineering - TODO

Before passing data to machine learning models, the nested NoSQL data must be transformed into a tabular format and converted into numerical vectors.

10. **`pipeline_enrichment/ml_preprocessing_pipeline.py`**
    * **Purpose:**
      1. **Flattening:** Normalizes the nested JSON from `jobs_enriched` into a flat Pandas DataFrame.
      2. **Feature Engineering:** Creates new structured variables and applies the Scikit-Learn `ColumnTransformer` (e.g., TF-IDF on descriptions, multi-hot encoding on skill arrays).
      3. **Topic Modeling:** Fits unsupervised models (like NMF) to automatically discover and label job clusters.
    * **Output:** Generates `data_vectorized.json` or writes the final engineered dataset back to the database.

## Phase 6: Post-Enrichment EDA & Insights - TODO

11. **`job_market_eda.ipynb` (Part 2)**
    * **Purpose:** Now that the data is fully enriched and topic-clustered, use the notebook to visualize skill co-occurrences, map geographical densities, and analyze the distribution of the newly discovered machine learning topics.

## Core Tech Stack:

Python 3.10+ & uv: The foundation of the pipeline, utilizing uv as a blazingly fast, Rust-based package manager to ensure strict dependency isolation and reproducible execution.

Google Cloud Storage (GCP): Acts as the immutable Cloud Data Lake, providing a secure, historical backup of all raw API payloads before any transformation occurs.

MongoDB Atlas: A fully managed NoSQL document database. It serves as the operational staging layer, chosen specifically for its ability to handle the highly polymorphic and unpredictable JSON schemas typical of third-party job APIs.

spaCy & FlashText: The hybrid NLP engine. FlashText provides lightning-fast exact string matching (Aho-Corasick automaton), while spaCy handles complex grammatical context and ambiguity in the German language.

Pandas & Scikit-Learn: The feature engineering backbone used to flatten nested NoSQL documents into 2D matrices, apply TF-IDF vectorization, and perform unsupervised topic discovery.
