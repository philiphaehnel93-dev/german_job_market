import base64
import json
import string
import urllib.parse
import urllib.request
from typing import Dict, List
from airflow.models import Variable

# Constants
KEYWORDS = ["Data Engineer", "Data Scientist"]

# Fetch API credentials securely from Airflow Variables
ADZUNA_APP_ID = Variable.get("ADZUNA_APP_ID")
ADZUNA_APP_KEY = Variable.get("ADZUNA_APP_KEY")

def extract_skills(text: str) -> List[str]:
    if not text:
        return []
    known_skills = [
        "python", "aws", "docker", "terraform", "ci/cd",
        "sql", "power_bi", "tableau", "dbt",
        "pytorch", "machine_learning", "scikit-learn",
        "spark", "kafka", "scala", "hadoop",
        "snowflake", "airflow", "fastapi", "postgresql", "redis",
        "bash", "git", "linux"
    ]
    known_multi_word = ["ci/cd", "machine learning", "scikit-learn"]

    text_lower = str.lower(text)
    clean_text = text_lower.translate(str.maketrans("", "", string.punctuation))
    words = clean_text.split()
    skill_set = set(skill for skill in words if skill in known_skills)

    for skill in known_multi_word:
        if f" {skill} " in text_lower.replace(",", " "):
            skill_set.add(skill)
    return list(skill_set)

def extract_seniority(title: str) -> str:
    for s in ["Junior", "Mid", "Senior", "Lead"]:
        if s.lower() in title.lower():
            return s
    return "Unspecified"

def fetch_adzuna(keywords: List[str], max_total_jobs: int = 2000) -> List[Dict]:
    results = []
    for keyword in keywords:
        page = 1
        while len(results) < max_total_jobs:
            params = {"app_id": ADZUNA_APP_ID, "app_key": ADZUNA_APP_KEY, "title_only": keyword}
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
                        
                        title = job.get("title", "Unknown")
                        description = job.get("description", "")
                        
                        results.append({
                            "origin_id": job.get("id"),
                            "source_api": "Adzuna",
                            "job_title": title,
                            "company_name": job.get("company", {}).get("display_name", "Unknown"),
                            "location": job.get("location", {}).get("display_name", "Unknown"),
                            "is_remote": False,
                            "seniority_level": extract_seniority(title),
                            "description": description,
                            "tech_stack": extract_skills(description)
                        })
                page += 1
            except Exception as e:
                print(f"Adzuna API Error on page {page} for '{keyword}': {e}")
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
