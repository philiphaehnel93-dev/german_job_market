import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

app = FastAPI(title="Airflow & MongoDB Bridge API")

# Allow requests from your Airflow custom view on port 8081
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Airflow Configuration (uses Docker internal service name)
AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://airflow-webserver:8080/api/v1")
AIRFLOW_AUTH = (
    os.getenv("AIRFLOW_USER", "airflow"),
    os.getenv("AIRFLOW_PASSWORD", "airflow"),
)

# MongoDB Configuration (uses Docker internal service name)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:admin@mongodb:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "job_market_db")

mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
jobs_collection = db["job_runs"]
clustered_collection = db["jobs_clustered"]

class TriggerJobRequest(BaseModel):
    job_name: str
    conf: Optional[Dict[str, Any]] = {}

def serialize_doc(doc: dict) -> dict:
    """Convert MongoDB ObjectId to string so FastAPI can return JSON."""
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

# 1. Combined Health Check (FastAPI + Airflow + MongoDB)
@app.get("/health")
async def health_check():
    # Check Airflow
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{AIRFLOW_BASE_URL}/health", timeout=3.0)
            airflow_status = "ok" if resp.status_code == 200 else "unhealthy"
        except httpx.RequestError:
            airflow_status = "unreachable"

    # Check MongoDB
    try:
        await mongo_client.admin.command("ping")
        mongo_status = "ok"
    except Exception:
        mongo_status = "unreachable"

    return {
        "fastapi": "ok",
        "airflow": airflow_status,
        "mongodb": mongo_status,
    }


# 2. Trigger an Airflow DAG & Save Metadata in MongoDB
@app.post("/jobs/{dag_id}/trigger")
async def trigger_and_store_job(dag_id: str, payload: TriggerJobRequest):
    async with httpx.AsyncClient(auth=AIRFLOW_AUTH) as client:
        resp = await client.post(
            f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns",
            json={"conf": payload.conf},
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        airflow_run = resp.json()

    # Store tracking document in MongoDB
    job_doc = {
        "job_name": payload.job_name,
        "dag_id": dag_id,
        "dag_run_id": airflow_run["dag_run_id"],
        "state": airflow_run["state"],
        "conf": payload.conf,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    insert_result = await jobs_collection.insert_one(job_doc)

    return {
        "mongo_id": str(insert_result.inserted_id),
        "dag_id": dag_id,
        "dag_run_id": airflow_run["dag_run_id"],
        "state": airflow_run["state"],
    }


# 3. Fetch Recent Jobs from MongoDB
@app.get("/jobs")
async def list_stored_jobs(limit: int = 20):
    cursor = jobs_collection.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
    jobs = await cursor.to_list(length=limit)
    return {"jobs": jobs}

@app.get("/jobs/clustered")
async def get_clustered_jobs(
    cluster_id: Optional[int] = Query(None, description="Filter by cluster ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
):
    query = {}
    if cluster_id is not None:
        query["cluster_id"] = cluster_id

    cursor = clustered_collection.find(query).limit(limit)
    docs = await cursor.to_list(length=limit)

    return {
        "count": len(docs),
        "items": [serialize_doc(doc) for doc in docs],
    }