import re
import unicodedata
from typing import Dict, Tuple, Optional, Any

from src.service.airtable_client.airtable_client import AirtableClient

class CVManager:
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

    def retrieve_all_cvs(self) -> Dict[str, Tuple[str, str]]:
        """
        Retrieve all CV records and return a dict:
        { record_id: (cv, summarized_cv) }
        """
        records = self.client.get_records()
        result: Dict[str, Tuple[str, str]] = {}
        for record in records.get("records", []):
            record_id = record["id"]
            fields = record.get("fields", {})
            cv = fields.get("cv", "")
            summarized_cv = fields.get("summarized_cv", "")
            result[record_id] = (cv, summarized_cv)
        return result

    def _get_middle_slice(self, text: Optional[str], length: int = 100) -> str:
        """
        Extract the middle `length` characters of normalized text.
        """
        normalized = self._normalize_text(text)
        total_length = len(normalized)
        if total_length <= length:
            return normalized
        start = (total_length - length) // 2
        return normalized[start:start + length]

    def find_unsummarized_cv(self, target_cv: str) -> Optional[Tuple[str, str]]:
        """
        Finds an unsummarized CV that matches the normalized middle portion of the input.
        """
        target_slice = self._get_middle_slice(target_cv)
        cvs = self.retrieve_all_cvs()
        for record_id, (cv, summarized) in cvs.items():
            if summarized:
                continue
            db_slice = self._get_middle_slice(cv)
            if target_slice in db_slice or db_slice in target_slice:
                return record_id, cv
        return None

    def find_cv(self, target_cv: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Finds a CV that matches the normalized middle portion of the input.
        Returns a tuple (record_id, cv, summarized_cv) if found, otherwise (None, None, None).
        """
        target_slice = self._get_middle_slice(target_cv)
        cvs = self.retrieve_all_cvs()
        for record_id, (cv, summarized_cv) in cvs.items():
            db_slice = self._get_middle_slice(cv)
            if target_slice in db_slice or db_slice in target_slice:
                return record_id, cv, summarized_cv
        return (None, None, None)

    def update_cv(self, record_tuple: Optional[Tuple[str, str]], new_summary: str) -> Any:
        """
        Update the given record (from find_unsummarized_cv) with new summarized_cv.
        """
        if not record_tuple:
            return None
        record_id, _ = record_tuple
        return self.client.update_record(record_id, {"summarized_cv": new_summary})

    def create_cv(self, cv: str, summarized_cv: str = "") -> Optional[str]:
        """
        Create a new CV record in the Airtable.
        Returns the ID of the created record.
        """
        fields = {
            "cv": cv,
            "summarized_cv": summarized_cv,
        }
        response = self.client.create_record(fields)
        return response.get("id") if response else None