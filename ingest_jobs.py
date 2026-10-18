import base64
import json
import string
import urllib.parse
import urllib.request
from typing import Dict, List
from google.cloud import storage

# Search target keywords
KEYWORDS = ["Data Engineer", "Data Scientist"]

# Adzuna Credentials
ADZUNA_APP_ID = "8641af88"
ADZUNA_APP_KEY = "626562fa763f5860a15d05798b0d1f3a"

# GCP Configuration
PROJECT_ID = "ml-pipeline-hu1tac"
BUCKET_NAME = f"{PROJECT_ID}-ml-data"

# ==========================================
# 1. Helper Functions
# ==========================================

def extract_skills(text: str) -> List[str]:
    """Extracts predefined tech stack keywords from a description string."""
    if not text:
        return []

    known_skills = [
        "python", "aws", "docker", "terraform", "ci/cd",
        "sql", "power_bi", "tableau", "dbt",
        "pytorch", "machine_learning", "scikit-learn",
        "spark", "kafka", "scala", "hadoop",
        "snowflake", "airflow",
        "fastapi", "postgresql", "redis",
        "bash", "git", "linux"
    ]
    known_multi_word = ["ci/cd", "machine learning", "scikit-learn"]

    text_lower = str.lower(text)
    clean_text = text_lower.translate(str.maketrans("", "", string.punctuation))
    words = clean_text.split()

    skill_set = set()

    for skill in words:
        if skill in known_skills:
            skill_set.add(skill)

    for skill in known_multi_word:
        if f" {skill} " in text_lower.replace(",", " "):
            skill_set.add(skill)

    return list(skill_set)


def extract_seniority(title: str) -> str:
    """Infers seniority from the job title."""
    for s in ["Junior", "Mid", "Senior", "Lead"]:
        if s.lower() in title.lower():
            return s
    return "Unspecified"


# ==========================================
# 2. API Ingestion Functions
# ==========================================

def fetch_adzuna(keywords: List[str], max_total_jobs: int = 2000) -> List[Dict]:
    """Fetches jobs from Adzuna API capped at max_total_jobs."""
    print(f"Fetching from Adzuna [Capped at {max_total_jobs} total]...")
    results = []

    for keyword in keywords:
        page = 1
        while len(results) < max_total_jobs:
            params = {
                "app_id": ADZUNA_APP_ID,
                "app_key": ADZUNA_APP_KEY,
                "title_only": keyword
            }
            url = f"https://api.adzuna.com/v1/api/jobs/de/search/{page}?{urllib.parse.urlencode(params)}"

            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    jobs = data.get("results", [])

                    if not jobs:
                        break

                    for job in jobs:
                        if len(results) >= max_total_jobs:
                            break

                        area = job.get("location", {}).get("area", [])
                        city = job.get("location", {}).get("display_name", "Unknown")
                        is_remote = False

                        if len(area) > 2:
                            city = f"{area[1]}, {area[-1]}"
                        if len(area) == 1 and area[0] in ["Germany", "Deutschland"]:
                            is_remote = True

                        description = job.get("description", "")

                        results.append({
                            "origin_id": str(job.get("id")),
                            "source_api": "Adzuna",
                            "job_title": job.get("title", "Unknown"),
                            "company_name": job.get("company", {}).get("display_name", "Unknown"),
                            "location": city,
                            "is_remote": is_remote,
                            "seniority_level": extract_seniority(job.get("title", "")),
                            "description": description,
                            "tech_stack": extract_skills(description)
                        })

                page += 1

            except Exception as e:
                print(f"Adzuna API Error on page {page} for '{keyword}': {e}")
                break

        if len(results) >= max_total_jobs:
            print(f"Reached cap of {max_total_jobs} jobs for Adzuna.")
            break

    return results


def fetch_jobsuche(keywords: List[str], max_total_jobs: int = 1000) -> List[Dict]:
    """Fetches jobs from AfA Jobsuche API capped at max_total_jobs."""
    print(f"Fetching from Jobsuche (AfA) [Capped at {max_total_jobs} total]...")
    results = []
    headers = {
        "Accept": "application/json",
        "X-API-Key": "jobboerse-jobsuche",
        "User-Agent": "Mozilla/5.0"
    }

    for keyword in keywords:
        page = 1
        while len(results) < max_total_jobs:
            params = {"was": keyword, "page": page, "size": 100}
            encoded_params = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
            search_url = f"https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v6/jobs?{encoded_params}"

            try:
                req = urllib.request.Request(search_url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    search_data = json.loads(response.read().decode("utf-8"))
                    ergebnisliste = search_data.get("ergebnisliste", [])

                    if not ergebnisliste:
                        break

                    for job in ergebnisliste:
                        if len(results) >= max_total_jobs:
                            break

                        refnr = job.get("referenznummer")
                        description = ""

                        if refnr:
                            encoded_refnr = base64.b64encode(refnr.encode("utf-8")).decode("utf-8")
                            details_url = f"https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobdetails/{encoded_refnr}"

                            try:
                                det_req = urllib.request.Request(details_url, headers=headers)
                                with urllib.request.urlopen(det_req) as det_res:
                                    det_data = json.loads(det_res.read().decode("utf-8"))
                                    description = det_data.get("stellenangebotsBeschreibung", "")
                            except Exception:
                                pass

                        title = job.get("stellenangebotsTitel", "Unknown")
                        locations = job.get("stellenlokationen", [])
                        city = (
                            locations[0].get("adresse", {}).get("ort", "Unknown")
                            if locations
                            else "Unknown"
                        )

                        results.append({
                            "origin_id": refnr,
                            "source_api": "Jobsuche",
                            "job_title": title,
                            "company_name": job.get("firma", "Unknown"),
                            "location": city,
                            "is_remote": job.get("homeofficemoeglich", False),
                            "seniority_level": extract_seniority(title),
                            "description": description,
                            "tech_stack": extract_skills(description)
                        })

                page += 1

            except Exception as e:
                print(f"Jobsuche API Error on page {page} for '{keyword}': {e}")
                break

        if len(results) >= max_total_jobs:
            print(f"Reached cap of {max_total_jobs} jobs for Jobsuche.")
            break

    return results


# ==========================================
# 3. Execution Pipeline & Cloud Storage Upload
# ==========================================

def main():
    unified_jobs = []

    # Ingestion phase
    unified_jobs.extend(fetch_adzuna(KEYWORDS, max_total_jobs=2000))
    unified_jobs.extend(fetch_jobsuche(KEYWORDS, max_total_jobs=1000))

    print(f"\nPipeline Complete. Ingested {len(unified_jobs)} total raw jobs.")

    if unified_jobs:
        print(f"Uploading raw payload to Cloud Storage bucket: {BUCKET_NAME}...")
        try:
            client = storage.Client(project=PROJECT_ID)
            bucket = client.bucket(BUCKET_NAME)

            blob = bucket.blob("raw_jobs_data.json")
            json_data = json.dumps(unified_jobs, ensure_ascii=False, indent=2)
            blob.upload_from_string(json_data, content_type="application/json")

            print(f"Successfully uploaded 'raw_jobs_data.json' to gs://{BUCKET_NAME}/")

        except Exception as e:
            print(f"Error uploading to Cloud Storage: {e}")


if __name__ == "__main__":
    main()
