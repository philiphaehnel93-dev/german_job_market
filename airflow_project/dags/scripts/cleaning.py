def clean_job_records(raw_jobs: list) -> list:
    """Cleans, normalizes, and standardizes raw job dictionaries."""
    cleaned_jobs = []
    for job in raw_jobs:
        job_title = job.get("job_title", "Unspecified")
        if not job_title or job_title == "Unknown":
            continue  # Skip invalid entries

        cleaned_job = {
            "origin_id": job.get("origin_id"),
            "source_api": job.get("source_api", "Unknown"),
            "job_title": job_title.strip(),
            "company_name": job.get("company_name", "Unknown").strip(),
            "location": job.get("location", "Unknown").strip(),
            "is_remote": bool(job.get("is_remote", False)),
            "seniority_level": job.get("seniority_level", "Unspecified"),
            "description": job.get("description", "").strip(),
            "tech_stack": job.get("tech_stack", [])
        }
        cleaned_jobs.append(cleaned_job)
        
    return cleaned_jobs
