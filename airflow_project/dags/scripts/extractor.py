import json
import re
from typing import List, Set
from dataclasses import dataclass

import spacy
from spacy.matcher import Matcher
from flashtext import KeywordProcessor
import pgeocode

@dataclass
class JobData:
    emails: List[str]
    location: dict
    tech_stack: List[str]

class HybridJobExtractor:
    def __init__(self, dict_path: str = "tech_dictionary.json"):
        # 1. Load the Tech Dictionary STRICTLY from the JSON file
        try:
            with open(dict_path, "r", encoding="utf-8") as f:
                tech_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"CRITICAL ERROR: The dictionary file '{dict_path}' is missing. The pipeline cannot start.")
        except json.JSONDecodeError:
            raise ValueError(f"CRITICAL ERROR: '{dict_path}' contains invalid JSON formatting. Please fix it before running.")

        # 2. Initialize FlashText for unambiguous terms
        self.tech_processor = KeywordProcessor(case_sensitive=False)
        self.tech_processor.add_keywords_from_dict(tech_dict)

        # 3. Initialize spaCy for ambiguous terms (German model)
        self.nlp = spacy.load("de_core_news_sm")
        self.matcher = Matcher(self.nlp.vocab)

        self.matcher.add("GOLANG", [[{"LOWER": "go", "POS": {"IN": ["PROPN", "FM", "NOUN"]}}]])
        self.matcher.add("REACT", [[{"LOWER": "react", "POS": {"IN": ["PROPN", "NOUN"]}}]])

        # Rule for "C" with contextual anchors
        self.matcher.add("C_LANG", [
            [{"TEXT": "C", "POS": {"IN": ["PROPN", "NOUN", "X", "FM"]}}],
            [{"TEXT": "C"}, {"LOWER": {"IN": ["entwickler", "programmierer", "developer", "engineer", "kenntnisse"]}}],
            [{"LOWER": "in"}, {"TEXT": "C"}]
        ])

        # Rule for "R" with data science contextual anchors
        self.matcher.add("R_LANG", [
            [{"TEXT": "R", "POS": {"IN": ["PROPN", "NOUN", "X", "FM"]}}],
            [{"TEXT": "R"}, {"LOWER": {"IN": ["entwickler", "programmierer", "developer", "engineer", "kenntnisse", "analyst", "scientist", "studio"]}}],
            [{"LOWER": "in"}, {"TEXT": "R"}]
        ])

        # 4. Initialize pgeocode for German postal codes
        self.geo_db = pgeocode.Nominatim('de')

    def extract_emails(self, text: str) -> List[str]:
        """Extracts emails using a standard robust regex."""
        pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        return list(set(re.findall(pattern, text)))

    def extract_location(self, text: str) -> dict:
        """Extracts remote status and coordinates, handling German negations."""
        # Check for remote words, but fail if "kein/keine" is immediately before it
        has_remote_word = bool(re.search(r'\b(remote|homeoffice|home-office)\b', text, re.IGNORECASE))
        has_negation = bool(re.search(r'\b(kein|keine)\s+(remote|homeoffice|home-office)\b', text, re.IGNORECASE))

        is_remote = has_remote_word and not has_negation
        location_data = {"remote": is_remote, "city": None, "lat": None, "lon": None}

        # Look for 5-digit German postal codes (PLZ)
        plz_match = re.search(r'\b([0-9]{5})\b', text)
        if plz_match:
            plz = plz_match.group(1)
            geo_info = self.geo_db.query_postal_code(plz)

            if not geo_info.empty and str(geo_info.place_name) != 'nan':
                # Take only the first neighborhood before the comma for clean data
                location_data["city"] = str(geo_info.place_name).split(',')[0].strip()
                location_data["lat"] = geo_info.latitude
                location_data["lon"] = geo_info.longitude

        return location_data

    def extract_tech_stack(self, text: str) -> List[str]:
        """Combines FlashText (fast) and spaCy (contextual) for tech extraction."""
        extracted: Set[str] = set()

        # Extract unambiguous terms via dictionary mapping
        extracted.update(self.tech_processor.extract_keywords(text))

        # Extract ambiguous/contextual terms via spaCy patterns
        doc = self.nlp(text)
        matches = self.matcher(doc)

        for match_id, start, end in matches:
            rule_name = self.nlp.vocab.strings[match_id]
            if rule_name == "GOLANG":
                extracted.add("Go")
            elif rule_name == "REACT":
                extracted.add("React")
            elif rule_name == "C_LANG":
                extracted.add("C")
            elif rule_name == "R_LANG":
                extracted.add("R")

        return list(extracted)

    def process_job_description(self, text: str) -> JobData:
        """Main pipeline method mapping raw text to structured JobData."""
        if not text or not isinstance(text, str):
            return JobData(emails=[], location={}, tech_stack=[])

        return JobData(
            emails=self.extract_emails(text),
            location=self.extract_location(text),
            tech_stack=self.extract_tech_stack(text)
        )
