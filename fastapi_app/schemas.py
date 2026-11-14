from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

# --- Request / Ingestion Schemas ---

class JobRecord(BaseModel):
    origin_id: str = Field(..., description="Unique identifier for the job")
    ml_model_version: str = Field(..., description="Version of the ML model used for clustering")

    # Allows dynamic fields (title, salary, etc.) to pass through validation
    model_config = ConfigDict(extra="allow")

class MLMetricsPayload(BaseModel):
    execution_date: str
    inertia: float
    silhouette_score: float
    model_version: str
    dag_run_id: Optional[str] = None
    task_id: Optional[str] = None


# --- Response Schemas ---

class ClusterGetResponse(BaseModel):
    count: int = Field(..., examples=[1527])
    items: List[Dict[str, Any]] = Field(
        ...,
        description="List of MongoDB Docs",
        examples=[[{"_id": "67338a...", "job_title": "Data Engineer"}]]
    )

class SecureJobsResponse(BaseModel):
    message: str
    service: Optional[str] = None
    count: int
    data: List[Dict[str, Any]]

class BulkSaveResponse(BaseModel):
    status: str
    matched: int = 0
    upserted: int = 0
    modified: int = 0

class MetricsSaveResponse(BaseModel):
    status: str
    id: str

class ErrorResponse(BaseModel):
    detail: str = Field(..., examples=["MongoDB connection timed out"])
