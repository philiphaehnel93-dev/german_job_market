"""
Shared configuration constants, salary baselines, and geographic lookup tables
for the German Tech Role Market Segmentation dashboard.
"""

from typing import Dict, List

import os

FASTAPI_BASE_URL: str = os.getenv(
    "FASTAPI_BASE_URL",
    "http://fastapi-service:8081/api/v1",
)

GERMANY_GEOJSON_URL: str = (
    "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/"
    "main/2_bundeslaender/4_niedrig.geo.json"
)
JOB_TITLES_DICT_PATH: str = "/app/tech_jobs_dictionary.json"
SMALL_CLUSTER_THRESHOLD: int = 15

ENTGELDATLAS_BASELINES: Dict[str, Dict[str, int]] = {
    "Data Engineer (43214)": {"p25": 55_200, "median": 72_400, "p75": 88_900},
    "Data Scientist (43104)": {"p25": 58_100, "median": 75_600, "p75": 92_300},
    "Software Engineer (43414)": {"p25": 52_400, "median": 68_800, "p75": 85_100},
}

CITIES_COORDS: Dict[str, Dict[str, float]] = {
    "Berlin": {"lat": 52.5200, "lon": 13.4050},
    "München": {"lat": 48.1351, "lon": 11.5820},
    "Munich": {"lat": 48.1351, "lon": 11.5820},
    "Hamburg": {"lat": 53.5511, "lon": 9.9937},
    "Frankfurt am Main": {"lat": 50.1109, "lon": 8.6821},
    "Frankfurt": {"lat": 50.1109, "lon": 8.6821},
    "Köln": {"lat": 50.9375, "lon": 6.9603},
    "Cologne": {"lat": 50.9375, "lon": 6.9603},
    "Stuttgart": {"lat": 48.7758, "lon": 9.1829},
    "Düsseldorf": {"lat": 51.2277, "lon": 6.7735},
    "Dusseldorf": {"lat": 51.2277, "lon": 6.7735},
    "Dortmund": {"lat": 51.5136, "lon": 7.4653},
    "Essen": {"lat": 51.4556, "lon": 7.0116},
    "Leipzig": {"lat": 51.3397, "lon": 12.3731},
    "Bremen": {"lat": 53.0793, "lon": 8.8017},
    "Dresden": {"lat": 51.0504, "lon": 13.7373},
    "Hannover": {"lat": 52.3759, "lon": 9.7320},
    "Nürnberg": {"lat": 49.4521, "lon": 11.0767},
    "Nuremberg": {"lat": 49.4521, "lon": 11.0767},
    "Duisburg": {"lat": 51.4344, "lon": 6.7623},
    "Bochum": {"lat": 51.4818, "lon": 7.2162},
    "Wuppertal": {"lat": 51.2562, "lon": 7.1508},
    "Bonn": {"lat": 50.7374, "lon": 7.0982},
    "Bielefeld": {"lat": 52.0302, "lon": 8.5325},
    "Mannheim": {"lat": 49.4875, "lon": 8.4660},
    "Karlsruhe": {"lat": 49.0069, "lon": 8.4037},
    "Wiesbaden": {"lat": 50.0782, "lon": 8.2397},
    "Münster": {"lat": 51.9607, "lon": 7.6261},
    "Aachen": {"lat": 50.7753, "lon": 6.0839},
    "Freiburg": {"lat": 47.9990, "lon": 7.8421},
    "Ulm": {"lat": 48.4011, "lon": 9.9876},
    "Bruchsal": {"lat": 49.1259, "lon": 8.5997},
}

GERMAN_STATES: List[str] = [
    "Baden-Württemberg",
    "Bayern",
    "Berlin",
    "Brandenburg",
    "Bremen",
    "Hamburg",
    "Hessen",
    "Mecklenburg-Vorpommern",
    "Niedersachsen",
    "Nordrhein-Westfalen",
    "Rheinland-Pfalz",
    "Saarland",
    "Sachsen",
    "Sachsen-Anhalt",
    "Schleswig-Holstein",
    "Thüringen",
]

# Maps raw role variants to a single canonical display role
JOB_ROLE_MAPPINGS: Dict[str, str] = {
    "ml engineer": "Machine Learning Engineer",
    "machine learning engineer": "Machine Learning Engineer",
    "software developer": "Software Engineer",
    "software engineer": "Software Engineer",
    "backend engineer": "Backend Developer",
    "backend developer": "Backend Developer",
    "devops engineer": "DevOps Engineer",
}

# Maps canonical lowercase role names to their job-title matching regex
ROLE_REGEX_PATTERNS: Dict[str, str] = {
    "business intelligence developer": r"business intelligence|\bbi\b|power bi|tableau|qlik",
    "analytics engineer": r"analytics engineer|dbt|data modeling",
    "data architect": r"data architect|datenarchitekt|enterprise architect",
    "data engineer": r"data engineer|data integration|big data|data platform|etl",
    "data scientist": r"data scientist|data science",
    "machine learning engineer": r"machine learning|\bml\b|mlops|ai engineer|computer vision|\bnlp\b",
    "data analyst": r"data analyst|datenanalyst|business analyst",
    "cloud engineer": r"cloud engineer|cloud architect|\baws\b|\bazure\b|\bgcp\b",
    "devops engineer": r"devops|\bsre\b|site reliability|platform engineer|infrastructure engineer",
    "backend developer": r"backend|back-end|back end",
    "software engineer": r"software|developer|programmer|frontend|fullstack|entwickler|\bswe\b",
    "forward deployed engineer": r"deploy forward|forward deployed",
}
# Maps cities (and common English/state aliases) to official GeoJSON Bundesland names
CITY_TO_STATE: Dict[str, str] = {
    "Berlin": "Berlin",
    "München": "Bayern",
    "Munich": "Bayern",
    "Nürnberg": "Bayern",
    "Nuremberg": "Bayern",
    "Augsburg": "Bayern",
    "Regensburg": "Bayern",
    "Erlangen": "Bayern",
    "Würzburg": "Bayern",
    "Bavaria": "Bayern",
    "Hamburg": "Hamburg",
    "Frankfurt am Main": "Hessen",
    "Frankfurt": "Hessen",
    "Wiesbaden": "Hessen",
    "Darmstadt": "Hessen",
    "Kassel": "Hessen",
    "Hesse": "Hessen",
    "Köln": "Nordrhein-Westfalen",
    "Cologne": "Nordrhein-Westfalen",
    "Düsseldorf": "Nordrhein-Westfalen",
    "Dusseldorf": "Nordrhein-Westfalen",
    "Dortmund": "Nordrhein-Westfalen",
    "Essen": "Nordrhein-Westfalen",
    "Duisburg": "Nordrhein-Westfalen",
    "Bochum": "Nordrhein-Westfalen",
    "Wuppertal": "Nordrhein-Westfalen",
    "Bonn": "Nordrhein-Westfalen",
    "Bielefeld": "Nordrhein-Westfalen",
    "Münster": "Nordrhein-Westfalen",
    "Aachen": "Nordrhein-Westfalen",
    "Stuttgart": "Baden-Württemberg",
    "Mannheim": "Baden-Württemberg",
    "Karlsruhe": "Baden-Württemberg",
    "Freiburg": "Baden-Württemberg",
    "Ulm": "Baden-Württemberg",
    "Bruchsal": "Baden-Württemberg",
    "Heidelberg": "Baden-Württemberg",
    "Leipzig": "Sachsen",
    "Dresden": "Sachsen",
    "Chemnitz": "Sachsen",
    "Bremen": "Bremen",
    "Hannover": "Niedersachsen",
    "Hanover": "Niedersachsen",
    "Braunschweig": "Niedersachsen",
    "Wolfsburg": "Niedersachsen",
    "Potsdam": "Brandenburg",
    "Kiel": "Schleswig-Holstein",
    "Lübeck": "Schleswig-Holstein",
    "Rostock": "Mecklenburg-Vorpommern",
    "Schwerin": "Mecklenburg-Vorpommern",
    "Mainz": "Rheinland-Pfalz",
    "Koblenz": "Rheinland-Pfalz",
    "Saarbrücken": "Saarland",
    "Magdeburg": "Sachsen-Anhalt",
    "Halle": "Sachsen-Anhalt",
    "Erfurt": "Thüringen",
    "Jena": "Thüringen",
    **{state: state for state in GERMAN_STATES},
}
