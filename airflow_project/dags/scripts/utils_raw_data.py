import base64
import json
import urllib.parse
import urllib.request
import os
from typing import Dict, List
from airflow.models import Variable

# Fetch API credentials securely from Airflow Variables
ADZUNA_APP_ID = Variable.get("ADZUNA_APP_ID")
ADZUNA_APP_KEY = Variable.get("ADZUNA_APP_KEY")

def get_keywords() -> List[str]:
    # 1. Initialize with a fallback to prevent UnboundLocalError
    all_jobs = ["Data Engineer", "Data Scientist"]
    
    json_path = os.path.join(os.path.dirname(__file__), "tech_data_jobs.json")
    
    # 2. Safely attempt to load the JSON file
    try:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                all_jobs = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load JSON file. Using fallback. Error: {e}")
        
    # 3. Safely fetch the Airflow variable (DAG parsing can sometimes block DB calls)
    try:
        requested_limit = int(Variable.get("JOB_KEYWORD_LIMIT", default_var=2))
    except Exception:
        requested_limit = 2
        
    # 4. Cap the limit and slice
    actual_limit = min(requested_limit, len(all_jobs))
    
    return all_jobs[:actual_limit]

KEYWORDS = get_keywords()

def extract_seniority(title: str) -> str:
    for s in ["Junior", "Mid", "Senior", "Lead"]:
        if s.lower() in title.lower():
            return s
    return "Unspecified"

def fetch_adzuna(keywords: List[str] = KEYWORDS, max_jobs_per_keyword: int = 1000) -> List[Dict]:
    results = []
    for keyword in keywords:
        page = 1
        keyword_count = 0
        while keyword_count < max_jobs_per_keyword:
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
                        if keyword_count >= max_jobs_per_keyword:
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
                            "tech_stack": []  # Let downstream NLP handle this
                        })
                        keyword_count += 1
                page += 1
            except Exception as e:
                print(f"Adzuna API Error on page {page} for '{keyword}': {e}")
                break
    return results

def fetch_jobsuche(keywords: List[str] = KEYWORDS, max_jobs_per_keyword: int = 1000) -> List[Dict]:
    print(f"Fetching from Jobsuche (AfA) [Capped at {max_jobs_per_keyword} per keyword]...")
    results = []
    headers = {
        "Accept": "application/json",
        "X-API-Key": "jobboerse-jobsuche",
        "User-Agent": "Mozilla/5.0"
    }

    for keyword in keywords:
        page = 1
        keyword_count = 0
        while keyword_count < max_jobs_per_keyword:
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
                        if keyword_count >= max_jobs_per_keyword:
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
                        city = locations[0].get("adresse", {}).get("ort", "Unknown") if locations else "Unknown"

                        results.append({
                            "origin_id": refnr,
                            "source_api": "Jobsuche",
                            "job_title": title,
                            "company_name": job.get("firma", "Unknown"),
                            "location": city,
                            "is_remote": job.get("homeofficemoeglich", False),
                            "seniority_level": extract_seniority(title),
                            "description": description,
                            "tech_stack": []  # Let downstream NLP handle this
                        })
                        keyword_count += 1
                page += 1

            except Exception as e:
                print(f"Jobsuche API Error on page {page} for '{keyword}': {e}")
                break

    return results
