from pymongo import MongoClient
import os

client = MongoClient(os.environ.get("MONGO_URI"))
db = client["job_market_db"]
sample_job = db["jobs_raw"].find_one({"description": {"$exists": True}})

# Print the length of the string, not the text itself
print(f"Total characters in description: {len(sample_job['description'])}")
print(sample_job.keys())
