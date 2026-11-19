"""
ML Feature Engineering Task
Flattens enriched job records, normalizes tech stacks, and builds a numeric feature matrix.
"""

import ast
import os
from time import perf_counter
from typing import List

import numpy as np
import pandas as pd
import requests
from airflow.operators.python import get_current_context
from sklearn.preprocessing import MultiLabelBinarizer

SENIORITY_MAP = {
    "Junior": 1,
    "Mid": 2,
    "Unspecified": 2,
    "Senior": 3,
    "Lead": 4,
}


def clean_tech_list(val) -> List[str]:
    """Safely extracts and normalizes tech names from Python lists, NumPy arrays, or stringified lists."""
    if isinstance(val, (list, tuple, np.ndarray)):
        return [str(t).strip().lower() for t in val if pd.notna(t) and str(t).strip()]
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(t).strip().lower() for t in parsed if str(t).strip()]
            except (ValueError, SyntaxError):
                pass
        return [
            s.strip().strip("'").strip('"').lower()
            for s in stripped.strip("[]").split(",")
            if s.strip()
        ]
    return []


def extract_is_remote(df: pd.DataFrame) -> pd.Series:
    """Extracts a clean 0/1 remote indicator from location_data or an existing is_remote column."""
    if "location_data" in df.columns:
        return df["location_data"].apply(
            lambda x: int(bool(x.get("remote", False))) if isinstance(x, dict) else 0
        )
    if "is_remote" in df.columns:
        return df["is_remote"].fillna(False).astype(int)
    return pd.Series(0, index=df.index, dtype=int)


def run_feature_engineering_local(input_path: str, output_path: str) -> str:
    """Flattens enriched job records and engineers numeric features for ML clustering."""
    start_time = perf_counter()
    print(f"📥 Loading enriched records from {input_path}...")
    df = pd.read_parquet(input_path)

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if df.empty:
        print("⚠️ No enriched jobs found to process.")
        pd.DataFrame().to_parquet(output_path, index=False)
        return output_path

    print("🧹 Flattening and engineering features...")

    # 1. Extract binary remote indicator
    is_remote_series = extract_is_remote(df)

    # 2. Safely normalize tech_stack (handles both Python lists and Parquet numpy.ndarray)
    if "tech_stack" in df.columns:
        clean_stacks = df["tech_stack"].apply(clean_tech_list)
    else:
        clean_stacks = pd.Series([[] for _ in range(len(df))], index=df.index)

    # 3. Multi-hot encode normalized tech stack
    mlb = MultiLabelBinarizer()
    tech_matrix = mlb.fit_transform(clean_stacks)
    tech_dummies = pd.DataFrame(
        tech_matrix,
        columns=[f"tech_{col}" for col in mlb.classes_],
        index=df.index,
    )

    # 4. Assemble numeric feature matrix
    feature_parts = [
        pd.DataFrame(
            {
                "is_remote": is_remote_series,
                "tech_stack_size": clean_stacks.apply(len),
            },
            index=df.index,
        ),
        tech_dummies,
    ]

    # 5. Ordinal seniority encoding (Junior=1 < Mid/Unspecified=2 < Senior=3 < Lead=4)
    sen_col = next((c for c in ("seniority_level", "level") if c in df.columns), None)
    if sen_col:
        seniority_score = df[sen_col].map(SENIORITY_MAP).fillna(2).astype(int)
        feature_parts.append(pd.DataFrame({"seniority_score": seniority_score}, index=df.index))

    # 6. Optional work_model one-hot encoding if present
    if "work_model" in df.columns:
        work_model_dummies = pd.get_dummies(
            df["work_model"].fillna("Unspecified").astype(str),
            prefix="model",
            dtype=int,
        )
        feature_parts.append(work_model_dummies)

    feature_df = pd.concat(feature_parts, axis=1).fillna(0)
    feature_df.columns = [str(col) for col in feature_df.columns]

    print(f"☁️ Saving engineered features ({feature_df.shape}) to {output_path}...")
    feature_df.to_parquet(output_path, index=False)
    print(f"✅ Successfully engineered and saved {len(feature_df)} feature rows.")

    # Log pipeline execution metrics to FastAPI
    context = get_current_context()
    run_id = context["dag_run"].run_id
    task_id = context["task"].task_id
    duration = round(perf_counter() - start_time, 2)

    try:
        response = requests.post(
            "http://airflow-fastapi:8081/api/v1/pipeline_metrics",
            json={
                "dag_id": "german_job_market_pipeline",
                "run_id": run_id,
                "task_id": task_id,
                "status": "success",
                "rows_processed": len(feature_df),
                "duration_seconds": duration,
                "execution_date": str(pd.Timestamp.utcnow()),
            },
            timeout=5,
        )
        response.raise_for_status()
        print(f"✅ Pipeline metrics successfully logged! (Status: {response.status_code})")
    except requests.exceptions.RequestException as exc:
        print(f"❌ Failed to log pipeline metrics: {exc}")
        raise

    return output_path
