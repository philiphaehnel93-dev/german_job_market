import os
import jwt
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

# Import your Pydantic models from schemas.py
from schemas import AirflowPayload, ClusterGetResponse, SaveResponse, ErrorResponse

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
clustered_collection = db["jobs_master"]


def serialize_doc(doc: dict) -> dict:
    """Convert MongoDB ObjectId to string so FastAPI can return JSON."""
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

@app.get("/")
async def root():
    return {"message": "Hello World from FastAPI, use /docs for endpoints"}

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

@app.get("/jobs/clustered", response_model=ClusterGetResponse,)
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

@app.post("/mongo/save", status_code=status.HTTP_201_CREATED,
    response_model=SaveResponse,
    summary="Save Airflow task output to MongoDB",
    tags=["MongoDB Ingestion"],
    responses={
        201: {"description": "Document successfully stored in MongoDB."},
        422: {"description": "Validation Error — missing or invalid Airflow payload fields."},
        500: {"model": ErrorResponse, "description": "Database connection or write failure."}
    })
async def save_to_mongo(payload: AirflowPayload):
    try:
        db = client[payload.database]
        collection = db[payload.collection]

        document = {
            "dag_id": payload.dag_id,
            "run_id": payload.run_id,
            "task_id": payload.task_id,
            "saved_at": datetime.now(timezone.utc),
            "data": payload.data
        }

        result = await collection.insert_one(document)
        return {
            "status": "inserted",
            "id": str(result.inserted_id),
            "collection": payload.collection
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB write failed: {str(e)}")

security = HTTPBearer()

# In production, load this from os.getenv("API_SECRET_KEY")
SECRET_KEY = os.getenv("API_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("API_SECRET_KEY environment variable is not set!")

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to validate the JWT in the Authorization header."""
    try:
        # FastAPI extracts the raw token from "Bearer <token>"
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# The Depends(verify_jwt) locks this endpoint
@app.get("/api/v1/jobs")
async def get_jobs(
    token_payload: dict = Depends(verify_jwt),
    cluster_id: Optional[int] = Query(None, description="Filter by cluster ID"),
    limit: int = Query(5000, ge=1, le=10000, description="Max records to return"),
):
    # If code reaches here, the token is valid.
    
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