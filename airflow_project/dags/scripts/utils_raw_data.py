import base64
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List
from airflow.models import Variable


def get_keywords() -> List[str]:
    all_jobs = ["Data Engineer", "Data Scientist"]
    json_path = os.path.join(os.path.dirname(__file__), "tech_jobs_dictionary.json")

    try:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                all_jobs = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load JSON file. Using fallback. Error: {e}")

    try:
        # Defaults to 10 if the variable is completely missing
        requested_limit = int(os.getenv("JOB_KEYWORD_LIMIT", 10))
    except Exception:
        requested_limit = 10

    actual_limit = min(requested_limit, len(all_jobs))
    print(f"Applied limit: {actual_limit}. Final keywords to query: {all_jobs[:actual_limit]}")
    return all_jobs[:actual_limit]


KEYWORDS = get_keywords()


def extract_seniority(title: str) -> str:
    for s in ["Junior", "Mid", "Senior", "Lead"]:
        if s.lower() in title.lower():
            return s
    return "Unspecified"


def _fetch_json_with_retry(
    req: urllib.request.Request,
    max_retries: int = 3,
    timeout: int = 15,
    context_label: str = "API",
) -> Dict:
    """Fetches JSON with exponential backoff on 403/429 rate-limit errors."""
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429) and attempt < max_retries:
                wait = (2 ** attempt) + random.uniform(1, 3)
                print(
                    f"⚠️ {context_label} rate-limited (HTTP {exc.code}). "
                    f"Retrying in {wait:.1f}s (attempt {attempt}/{max_retries})..."
                )
                time.sleep(wait)
            else:
                raise


def fetch_adzuna(keywords: List[str] = KEYWORDS, max_jobs_per_keyword: int = 1000) -> List[Dict]:
    # SAFELY FETCH VARIABLES HERE: Only runs when the task executes
    app_id = Variable.get("ADZUNA_APP_ID")
    app_key = Variable.get("ADZUNA_APP_KEY")

    results = []
    for keyword in keywords:
        page = 1
        keyword_count = 0
        print(f"\n--- Starting Adzuna fetch for: '{keyword}' ---")

        while keyword_count < max_jobs_per_keyword:
            params = {
                "app_id": app_id,
                "app_key": app_key,
                "title_only": keyword,
                "results_per_page": 200,
            }
            url = f"https://api.adzuna.com/v1/api/jobs/de/search/{page}?{urllib.parse.urlencode(params)}"
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                data = _fetch_json_with_retry(
                    req,
                    context_label=f"Adzuna page {page} ('{keyword}')",
                )
                jobs = data.get("results", [])

                if not jobs:
                    print(f"Adzuna: No more jobs available for '{keyword}' after page {page - 1}. Moving to next keyword.")
                    break

                print(f"Adzuna: Fetched {len(jobs)} jobs on page {page} for '{keyword}' (Total so far: {keyword_count + len(jobs)})")

                for job in jobs:
                    if keyword_count >= max_jobs_per_keyword:
                        print(f"Adzuna: Hit the max limit of {max_jobs_per_keyword} for '{keyword}'.")
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
                        "tech_stack": [],
                    })
                    keyword_count += 1
                page += 1
            except Exception as e:
                print(f"Adzuna API Error on page {page} for '{keyword}': {e}")
                break
    return results


def fetch_jobsuche(keywords: List[str] = KEYWORDS, max_jobs_per_keyword: int = 1000) -> List[Dict]:
    print(f"\nFetching from Jobsuche (AfA) [Capped at {max_jobs_per_keyword} per keyword]...")
    results = []
    headers = {
        "Accept": "application/json",
        "X-API-Key": "jobboerse-jobsuche",
        "User-Agent": "Mozilla/5.0",
    }

    for keyword in keywords:
        page = 1
        keyword_count = 0
        print(f"\n--- Starting Jobsuche fetch for: '{keyword}' ---")

        while keyword_count < max_jobs_per_keyword:
            params = {"was": keyword, "page": page, "size": 200}
            encoded_params = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
            search_url = f"https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v6/jobs?{encoded_params}"

            try:
                req = urllib.request.Request(search_url, headers=headers)
                search_data = _fetch_json_with_retry(
                    req,
                    max_retries=3,
                    context_label=f"Jobsuche page {page} ('{keyword}')",
                )
                ergebnisliste = search_data.get("ergebnisliste", [])

                if not ergebnisliste:
                    print(f"Jobsuche: No more jobs available for '{keyword}' after page {page - 1}. Moving to next keyword.")
                    break

                print(f"Jobsuche: Fetched {len(ergebnisliste)} jobs on page {page} for '{keyword}' (Total so far: {keyword_count + len(ergebnisliste)})")

                for job in ergebnisliste:
                    if keyword_count >= max_jobs_per_keyword:
                        print(f"Jobsuche: Hit the max limit of {max_jobs_per_keyword} for '{keyword}'.")
                        break

                    refnr = job.get("referenznummer")
                    description = ""

                    if refnr:
                        encoded_refnr = base64.b64encode(refnr.encode("utf-8")).decode("utf-8")
                        details_url = f"https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobdetails/{encoded_refnr}"
                        try:
                            det_req = urllib.request.Request(details_url, headers=headers)
                            with urllib.request.urlopen(det_req, timeout=10) as det_res:
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
                        "tech_stack": [],
                    })
                    keyword_count += 1
                page += 1

            except Exception as e:
                print(f"Jobsuche API Error on page {page} for '{keyword}': {e}")
                break

        # Pause between keywords to avoid triggering Jobsuche WAF burst blocks
        time.sleep(random.uniform(2, 5))

    return results
