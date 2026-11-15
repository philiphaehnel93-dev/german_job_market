import os
import requests
from airflow.operators.python import get_current_context
import pandas as pd

def run_feature_engineering_local(input_path: str, output_path: str) -> str:
    """Flattens enriched job records and multi-hot encodes tech stack for ML clustering."""
    print(f"📥 Loading enriched records from {input_path}...")
    df = pd.read_parquet(input_path)

    if df.empty:
        print("⚠️ No enriched jobs found to process.")
        # Create an empty parquet file to prevent downstream task failures
        directory = os.path.dirname(output_path)
        os.makedirs(directory, exist_ok=True)
        pd.DataFrame().to_parquet(output_path, index=False)
        return output_path

    print("🧹 Flattening and encoding features...")

    # 1. Flatten location data fields safely (extracting 'remote' boolean as integer 0/1)
    df['is_remote'] = df['location_data'].apply(
        lambda x: int(x.get('remote', False)) if isinstance(x, dict) else 0
    )

    # 2. Multi-hot encode the 'tech_stack' list into binary dummy columns
    if 'tech_stack' in df.columns:
        # Explode lists into rows, generate dummy columns, then group back by original index and sum
        tech_dummies = df['tech_stack'].explode().str.get_dummies().groupby(level=0).sum()
    else:
        tech_dummies = pd.DataFrame(index=df.index)

    # 3. Combine binary and numerical features into a clean feature matrix
    feature_df = pd.concat([df[['is_remote']], tech_dummies], axis=1).fillna(0)
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    print(f"☁️ Saving engineered features to {output_path}...")
    feature_df.to_parquet(output_path, index=False)
    print(f"✅ Successfully engineered and saved {len(feature_df)} feature rows.")

    # Get Airflow context to log metrics to FastAPI endpoint
    context = get_current_context()
    run_id = context['dag_run'].run_id
    task_id = context['task'].task_id

    try:
        # 1. Assign the request to the 'response' variable
        response = requests.post(
            "http://airflow-fastapi:8081/api/v1/pipeline_metrics",
            json={
                "dag_id": "german_job_market_pipeline",
                "run_id": context['dag_run'].run_id,
                "task_id": context['task'].task_id,
                "status": "success",
                "rows_processed": len(feature_df),  # 2. Use feature_df here
                "duration_seconds": 0.0,            # 3. Set duration to 0.0
                "execution_date": str(pd.Timestamp.utcnow())
            },
            timeout=5
        )
        # Now this will work perfectly
        response.raise_for_status()
        print(f"✅ Pipeline metrics successfully logged! (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Failed to log pipeline metrics: {e}")
        raise
    return output_path
