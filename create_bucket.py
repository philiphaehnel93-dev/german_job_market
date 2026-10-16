import sys
from google.cloud import storage
from google.cloud.exceptions import Conflict, NotFound

def setup_gcp_bucket(project_id: str, bucket_name: str, location: str = "europe-west1"):
    client = storage.Client(project=project_id)
    try:
        bucket = client.get_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except NotFound:
        print(f"Creating new bucket: {bucket_name}...")
        bucket = client.bucket(bucket_name)
        bucket.storage_class = "STANDARD"
        try:
            bucket = client.create_bucket(bucket, location=location)
            print(f"Successfully created: {bucket.name} in {location}")
        except Conflict:
            print("Error: Bucket name is already taken globally.")
            sys.exit(1)
    return bucket

if __name__ == "__main__":
    PROJECT_ID = "ml-pipeline-hu1tac"
    BUCKET_NAME = f"{PROJECT_ID}-ml-data"
    setup_gcp_bucket(PROJECT_ID, BUCKET_NAME)
