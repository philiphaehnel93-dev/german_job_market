# Data Flow & Machine Learning Architecture

Data lifecycle for the Job Market pipeline for a job posting from raw API ingestion through NLP enrichment, Exploratory Data Analysis (EDA), and finally to Machine Learning modeling.

---

## 1. Ingestion Phase: Raw Extraction to Data Lake

*   **Action:** Job postings are queried from the Adzuna and Jobsuche APIs.
*   **Storage:** The aggregated, raw JSON payload is uploaded directly to a Google Cloud Storage (GCP) bucket.
*   **Purpose:** This serves as an immutable Data Lake, ensuring we always have a pristine backup of the original data before any transformations occur.

## 2. Staging Phase: Data Lake to Operational Database

*   **Action:** The raw JSON is pulled from the GCP bucket and loaded into the `jobs_raw` collection in MongoDB Atlas.
*   **Why MongoDB Atlas?** Job API data is highly polymorphic and semi-structured. Some jobs have nested location data, complex requirements arrays, and many fields are missing entirely. A NoSQL document database like MongoDB handles this variable schema naturally, preventing the rigid pipeline failures that would occur in a traditional SQL relational database during early exploratory phases.

## 3. Enrichment Phase: Hybrid NLP Extraction

*   **Action:** The pipeline reads raw descriptions from MongoDB and extracts structured entities (tech stack, emails, location) using a multi-layered NLP approach, writing the results to the `jobs_enriched` collection.
*   **Why FlashText?** Used for **unambiguous terms** (e.g., "JavaScript", "Docker"). FlashText uses an Aho-Corasick automaton, meaning it can search a document for 10,000 keywords in the exact same time it takes to search for one. It is exponentially faster and more scalable than Regex for large dictionaries.
*   **Why spaCy?** Used for **contextual ambiguity**. FlashText cannot distinguish between the programming language "C" and a random typo. We use spaCy's German model (`de_core_news_sm`) to analyze sentence structure, Part-of-Speech (POS) tags, and surrounding anchor words (e.g., "Entwickler") to accurately extract tricky terms like "C", "R", and "Go".
*   **Do we use Regex?** Yes, but only for globally strict, deterministic formats where context doesn't matter—specifically, extracting email addresses and identifying negative geographical constraints (e.g., using negative lookbehind for "keine remote").

## 4. Exploratory Data Analysis (EDA) & Flattening

*   **Environment:** Jupyter Notebook (`.ipynb`).
*   **When to use it:** This step acts as the interactive bridge between the automated NLP pipeline and the Machine Learning models. It is executed after the Enrichment phase once the data has structure and entities.
*   **Action:** The enriched data is pulled from MongoDB (or loaded from the local `data_enriched.json` export) into a Pandas DataFrame. The nested fields (like `location.city`) are flattened into distinct columns.
*   **Visual Exploration:** The notebook is used to iteratively generate visualizations (e.g., bar charts for the most in-demand technologies, distribution of remote vs. on-site roles, and geographic mapping using the extracted coordinates) to identify market trends and validate data quality.


---
## TODO

*   **  ML Preparation (Vectorization):** As a separate script the data is preprocessed for mathematical modeling. The `tech_stack` array is One-Hot Encoded (e.g., `has_python: 1`, `has_aws: 0`), and the full job `description` is processed using **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert natural language into numerical vectors.

## 5. Machine Learning Models (Automated Topic Discovery)

Once the data is flattened and vectorized, we apply unsupervised learning to discover patterns in the job market, utilizing automated topic labeling to bypass manual cluster analysis:

*   **Topic Modeling with BERTopic (Automated Labeling):** Instead of standard distance-based clustering algorithms that require manual inspection to deduce cluster meanings, we can use BERTopic to automatically generate human-readable topic labels. BERTopic uses transformer-based embeddings (like BERT) to understand semantic meaning and clusters documents based on context. It then applies a class-based TF-IDF (c-TF-IDF) calculation within clusters to prioritize representative words, ensuring that terms strongly associated with a specific cluster form highly interpretable labels automatically.
*   **Scikit-Learn Topic Modeling (NMF / LDA):** As a lightweight, API-consistent alternative, we can utilize Non-Negative Matrix Factorization (NMF) or Latent Dirichlet Allocation (LDA). NMF decomposes high-dimensional TF-IDF vectors into lower-dimensional, non-negative representations to extract an additive model of the topic structure. LDA builds a topic-per-document and words-per-topic model modeled as Dirichlet distributions to discover abstract topics. Both inherently output the most descriptive keywords, providing an automated representation of the topics without manual labeling.
*   **Cosine Similarity (Optional/Future):** By calculating the distance between the TF-IDF vector of a user's resume and the job vectors, we can build a personalized job recommendation engine.

---

## 6. Next Steps: Containerization & Orchestration

To transition this architecture from individual Python scripts into a production-grade, automated system, the following technologies will be integrated:

*   **Docker (Microservices Architecture):** Each phase of the pipeline (Ingestion, Staging, Enrichment) will be packaged into its own isolated Docker container. This ensures environment consistency across development and production, eliminates dependency conflicts (e.g., spaCy model installations), and allows each component to scale independently.
*   **Apache Airflow (Workflow Orchestration):** Manual script execution will be replaced by Airflow. We will define Directed Acyclic Graphs (DAGs) to automate the execution of our Docker containers. Airflow will manage the scheduling (e.g., running the ingestion daily at midnight), handle automatic retries upon failure, and provide a UI to monitor the health and logs of the entire ETL lifecycle.
