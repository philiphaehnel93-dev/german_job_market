# Job Market Data Pipeline

This project contains a data pipeline to ingest scraped job data from a Google Cloud Storage bucket into a MongoDB Atlas cluster, followed by data cleaning (hash-based deduplication) and Exploratory Data Analysis (EDA) using a Jupyter Notebook.

## Prerequisites

- Python 3.10+
- `uv` (Python package manager)
- Homebrew (for macOS users)
- Google Cloud CLI (`gcloud`) installed and configured
- MongoDB Atlas account

## Setup Instructions

### 1. Install System Dependencies

Install the MongoDB Atlas CLI using Homebrew:
```bash
brew install mongodb-atlas-cli
```

### 2. Configure MongoDB Atlas

Authenticate the Atlas CLI:
```bash
atlas setup
```
Select **API Keys** when prompted and provide your Public and Private keys generated from the Atlas console. Choose **plaintext** for the output format.

If you are creating a new cluster for this project:
```bash
atlas cluster create ml-pipeline-cluster --tier M0 --provider AWS --region EU_CENTRAL_1
```
*(Note: If you used the interactive setup wizard, your cluster might be named something like `Cluster0` or `Cluster47025`. Check available clusters using `atlas clusters list`.)*

Create a database user for your application:
```bash
atlas dbusers create --username admin --password <YourSecurePassword> --role readWriteAnyDatabase
```

### 3. Connecting to an Existing MongoDB Atlas Cluster

If your partner has already created the cluster and project on their account, they do not need to run the `atlas cluster create` command. Instead, they should:

1. Request that you (the project owner) add their MongoDB Atlas account email to the project as a user via the Atlas web UI (Project -> Access Manager).
2. Alternatively, they can generate their own API keys on their Atlas account, authenticate with `atlas setup`, and list the available clusters in the shared project using `atlas clusters list`.
3. Ensure their IP address is whitelisted in the Atlas Network Access settings.
4. They can then retrieve the connection string using:
   ```bash
   atlas clusters connectionStrings describe <ClusterName>
   ```

### 4. Setup Python Environment

Install dependencies using `uv`:
```bash
uv pip install pymongo pandas scikit-learn matplotlib seaborn python-dotenv google-cloud-storage jupyter
```

### 5. Configure Environment Variables

Create a `.env` file in the `scripts` directory (or wherever you run the notebook) and add your MongoDB connection string:
```text
MONGO_URI="mongodb+srv://admin:<YourSecurePassword>@<cluster-url>.mongodb.net/?retryWrites=true&w=majority"
```
Also, ensure the environment variable is set in your terminal session for standard scripts:
```bash
export MONGO_URI="mongodb+srv://admin:<YourSecurePassword>@<cluster-url>.mongodb.net/?retryWrites=true&w=majority"
```

### 6. Run the Pipeline

1. **Ingest Data:** Run `gcs_to_mongo.py` to transfer data from the GCS bucket to MongoDB.
   ```bash
   python gcs_to_mongo.py
   ```
2. **Data Analysis:** Open `job_market_eda.ipynb` in Jupyter Notebook or VS Code to perform data cleaning (MD5 hash deduplication based on company and description) and generate EDA visualizations.