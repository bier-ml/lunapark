import re
import unicodedata
from typing import Dict, Tuple, Optional, Any, List

from src.service.airtable_client.airtable_client import AirtableClient

class PairManager:
    def __init__(self, airtable_client: AirtableClient):
        self.client = airtable_client

    def _normalize_text(self, text: Optional[str]) -> str:
        """
        Normalize text by:
        - Lowercasing
        - Keeping only letters, digits, punctuation, and whitespace
        - Stripping extra spaces
        """
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text)
        text = text.lower()
        # Keep only letters, numbers, punctuation, and whitespace
        text = ''.join(
            ch for ch in text
            if unicodedata.category(ch).startswith(('L', 'N', 'P', 'Z'))
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def save_pair_result(self, cv_id: str, vacancy_id: str, comment: str, score: float) -> Optional[str]:
        """
        Save a new CV-vacancy pair result.
        Returns the ID of the created record.
        """
        fields = {
            "pairId": cv_id + "_" + vacancy_id,
            "cv": cv_id,
            "vacancy": vacancy_id,
            "comment": comment,
            "score": score
        }
        # AirtableClient will wrap this in {"fields": fields}
        print(f"Saving pair result: {fields}")
        response = self.client.create_record(fields)
        return response.get("id") if response else None

    def update_pair_result(self, pair_id: str, comment: str, score: float) -> Any:
        """
        Update an existing CV-vacancy pair result.
        """
        fields = {
            "comment": comment,
            "score": score
        }
        return self.client.update_record(pair_id, fields)

    def get_pair_result(self, cv_id: str, vacancy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the result for a specific CV-vacancy pair.
        Returns a dictionary with the pair details if found, None otherwise.
        """
        records = self.client.get_records()
        for record in records.get("records", []):
            fields = record.get("fields", {})
            if fields.get("cv") == cv_id and fields.get("vacancy") == vacancy_id:
                return {
                    "id": record["id"],
                    "comment": fields.get("comment", ""),
                    "score": fields.get("score", 0.0)
                }
        return None

    def get_all_results_for_cv(self, cv_id: str) -> List[Dict[str, Any]]:
        """
        Get all results for a specific CV.
        Returns a list of dictionaries with pair details.
        """
        records = self.client.get_records()
        results = []
        for record in records.get("records", []):
            fields = record.get("fields", {})
            if fields.get("cv") == cv_id:
                results.append({
                    "id": record["id"],
                    "vacancy_id": fields.get("vacancy", ""),
                    "comment": fields.get("comment", ""),
                    "score": fields.get("score", 0.0)
                })
        return results

    def get_all_results_for_vacancy(self, vacancy_id: str) -> List[Dict[str, Any]]:
        """
        Get all results for a specific vacancy.
        Returns a list of dictionaries with pair details.
        """
        records = self.client.get_records()
        results = []
        for record in records.get("records", []):
            fields = record.get("fields", {})
            if fields.get("vacancy") == vacancy_id:
                results.append({
                    "id": record["id"],
                    "cv_id": fields.get("cv", ""),
                    "comment": fields.get("comment", ""),
                    "score": fields.get("score", 0.0)
                })
        return results