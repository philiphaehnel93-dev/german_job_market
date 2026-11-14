import os
import jwt
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
from pymongo.errors import ConnectionFailure
from schemas import MLMetricsPayload, PipelineMetricPayload, JobRecord, ClusterGetResponse, SecureJobsResponse, BulkSaveResponse, MetricsSaveResponse, ErrorResponse

# Import the optimized Pydantic models
from schemas import (
    JobRecord, MLMetricsPayload, ClusterGetResponse,
    SecureJobsResponse, BulkSaveResponse, MetricsSaveResponse, ErrorResponse
)

from contextlib import asynccontextmanager

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Airflow & MongoDB Bridge API")

# Allow requests from your Airflow custom view on port 8081
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Airflow Configuration
AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://airflow-webserver:8080/api/v1")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:admin@mongodb:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "job_market_db")

mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
jobs_collection = db["job_runs"]
clustered_collection = db["jobs_master"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs once when the FastAPI server starts
    logger.info("Building database indexes...")

    # 1. Speeds up bulk upserts and prevents duplicate jobs
    await clustered_collection.create_index("origin_id", unique=True)

    # 2. Speeds up Streamlit dashboard queries filtering by cluster
    await clustered_collection.create_index("cluster_id")

    # 3. Auto-deletes old API audit logs after 30 days (2592000 seconds)
    await db["api_access_logs"].create_index("accessed_at", expireAfterSeconds=2592000)

    yield  # Server is now running and accepting requests

# Update your app initialization to use the lifespan
app = FastAPI(title="Airflow & MongoDB Bridge API", lifespan=lifespan)


def serialize_doc(doc: dict) -> dict:
    """Convert MongoDB ObjectId to string so FastAPI can return JSON."""
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


@app.get("/")
async def root():
    return {"message": "Hello World from FastAPI, use /docs for endpoints"}


@app.get("/health")
async def health_check():
    """Combined Health Check (FastAPI + Airflow + MongoDB)"""
    # Check Airflow
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{AIRFLOW_BASE_URL}/health", timeout=3.0)
            airflow_status = "ok" if resp.status_code == 200 else "unhealthy"
        except httpx.RequestError:
            airflow_status = "unreachable"

    # Check MongoDB (Catching specific PyMongo exception instead of bare Exception)
    try:
        await mongo_client.admin.command("ping")
        mongo_status = "ok"
    except ConnectionFailure:
        mongo_status = "unreachable"

    return {
        "fastapi": "ok",
        "airflow": airflow_status,
        "mongodb": mongo_status,
    }


# --- AIRFLOW INGESTION ENDPOINTS ---

@app.post("/api/v1/jobs/bulk", response_model=BulkSaveResponse, tags=["MongoDB Ingestion"])
async def bulk_save_jobs(jobs: List[JobRecord]):
    """Handles bulk upserts from the Airflow clustering task."""
    if not jobs:
        return {"status": "ignored", "matched": 0, "upserted": 0, "modified": 0}

    operations = [
        UpdateOne({"origin_id": j.origin_id}, {"$set": j.model_dump()}, upsert=True)
        for j in jobs
    ]

    if operations:
        try:
            result = await clustered_collection.bulk_write(operations)
            return {
                "status": "success",
                "matched": result.matched_count,
                "upserted": result.upserted_count,
                "modified": result.modified_count
            }
        except Exception as e:
            logger.error(f"Bulk write error: {e}")
            raise HTTPException(status_code=500, detail="Database write failed")

    return {"status": "no_operations", "matched": 0, "upserted": 0, "modified": 0}


@app.post("/api/v1/ml_metrics", response_model=MetricsSaveResponse, tags=["MongoDB Ingestion"])
async def save_ml_metrics(metrics: MLMetricsPayload):
    """Saves model evaluation metrics from Airflow."""
    try:
        result = await db["ml_metrics_history"].insert_one(metrics.model_dump())
        return {"status": "success", "id": str(result.inserted_id)}
    except Exception as e:
        logger.error(f"Metrics write error: {e}")
        raise HTTPException(status_code=500, detail="Database write failed")


# --- DATA RETRIEVAL ENDPOINTS ---

@app.get("/jobs/clustered", response_model=ClusterGetResponse, tags=["Public Data"])
async def get_clustered_jobs(
    cluster_id: Optional[int] = Query(None, description="Filter by cluster ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
):
    """Unprotected endpoint for clustered jobs."""
    query = {}
    if cluster_id is not None:
        query["cluster_id"] = cluster_id

    cursor = clustered_collection.find(query).limit(limit)
    docs = await cursor.to_list(length=limit)

    logger.info(f"Retrieved {len(docs)} clustered jobs (Unprotected)")

    return {
        "count": len(docs),
        "items": [serialize_doc(doc) for doc in docs],
    }


# --- SECURE DATA RETRIEVAL ---

security = HTTPBearer()
SECRET_KEY = os.getenv("FASTAPI_SECRET_KEY")
if not SECRET_KEY:
    logger.warning("API_SECRET_KEY environment variable is not set! JWT authentication will fail.")

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to validate the JWT in the Authorization header."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/api/v1/jobs", response_model=SecureJobsResponse, tags=["Secure Data"])
async def get_jobs_secure(
    token_payload: dict = Depends(verify_jwt),
    cluster_id: Optional[int] = Query(None, description="Filter by cluster ID"),
    limit: int = Query(5000, ge=1, le=10000, description="Max records to return"),
):
    """Protected endpoint requiring a valid JWT Bearer token."""
    # Synchronous Audit Logging
    log_entry = {
        "service_identity": token_payload.get("service", "unknown"),
        "endpoint": "/api/v1/jobs",
        "query_params": {"cluster_id": cluster_id, "limit": limit},
        "accessed_at": datetime.now(timezone.utc)
    }
    await db["api_access_logs"].insert_one(log_entry)

    # Fetch Data
    query = {}
    if cluster_id is not None:
        query["cluster_id"] = cluster_id

    cursor = clustered_collection.find(query).limit(limit)
    docs = await cursor.to_list(length=limit)

    return {
        "message": "Authentication successful",
        "service": token_payload.get("service"),
        "count": len(docs),
        "data": [serialize_doc(doc) for doc in docs]
    }


# --- STATS ENDPOINTS ---

@app.get("/stats/ingestion", response_model=List[Dict[str, Any]], tags=["Admin"])
async def get_ingestion_stats(limit: int = Query(10, description="Number of recent runs to fetch")):
    """Returns raw dictionary history for ingestion stats."""
    cursor = db["ingestion_stats"].find({}, {"_id": 0}).sort("run_date", -1).limit(limit)
    stats = await cursor.to_list(length=limit)
    return stats


@app.get("/stats/api_access", response_model=List[Dict[str, Any]], tags=["Admin"])
async def get_api_access_logs(limit: int = Query(10, description="Recent access logs")):
    """Returns raw dictionary history for API access logs."""
    cursor = db["api_access_logs"].find({}, {"_id": 0}).sort("accessed_at", -1).limit(limit)
    logs = await cursor.to_list(length=limit)
    return logs

@app.get("/api/v1/ml_metrics", tags=["Secure Data"])
async def get_ml_metrics(
    token_payload: dict = Depends(verify_jwt),
    limit: int = Query(50, description="Recent metrics to fetch")
):
    """Protected endpoint for Streamlit to fetch ML performance history."""
    cursor = db["ml_metrics_history"].find({}, {"_id": 0}).sort("execution_date", -1).limit(limit)
    metrics = await cursor.to_list(length=limit)
    return metrics

@app.post("/api/v1/pipeline_metrics", tags=["MongoDB Ingestion"])
async def save_pipeline_metrics(metric: PipelineMetricPayload):
    """Saves Airflow task execution statistics (rows processed, duration)."""
    try:
        await db["ingestion_stats"].insert_one(metric.model_dump())
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Pipeline metrics write error: {e}")
        raise HTTPException(status_code=500, detail="Database write failed")
