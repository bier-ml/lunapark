import re
from typing import Dict, Tuple, Optional, Any
import unicodedata
from src.service.airtable_client.airtable_client import AirtableClient


class VacancyManager:
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

    def retrieve_all_vacancies(self) -> Dict[str, Tuple[str, str]]:
        """
        Retrieve all vacancy records and return a dict:
        { record_id: (vacancy, summarized_vacancy) }
        """
        records = self.client.get_records()
        result: Dict[str, Tuple[str, str]] = {}
        for record in records.get("records", []):
            record_id = record["id"]
            fields = record.get("fields", {})
            vacancy = fields.get("vacancy", "")
            summarized_vacancy = fields.get("summarized_vacancy", "")
            result[record_id] = (vacancy, summarized_vacancy)
        return result

    def _get_middle_slice(self, text: str, length: int = 100) -> str:
        """
        Extract the middle `length` characters of normalized text.
        """
        normalized = self._normalize_text(text)
        total_length = len(normalized)
        if total_length <= length:
            return normalized
        start = (total_length - length) // 2
        return normalized[start:start + length]
    
    def find_vacancy(self, target_vacancy: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Finds a vacancy that matches the normalized middle portion of the input.
        Returns a tuple (record_id, vacancy, summarized_vacancy) if found, otherwise None.
        """
        target_slice = self._get_middle_slice(target_vacancy)
        vacancies = self.retrieve_all_vacancies()
        for record_id, (vacancy, summarized_vacancy) in vacancies.items():
            db_slice = self._get_middle_slice(vacancy)
            if target_slice in db_slice or db_slice in target_slice:
                return record_id, vacancy, summarized_vacancy
        return (None, None, None)

    def update_vacancy(self, record_tuple: Optional[Tuple[str, str]], new_summary: str) -> Any:
        """
        Update the given record (from find_unsummarized_vacancy) with new summarized_vacancy.
        """
        if not record_tuple:
            return None
        record_id, _ = record_tuple
        return self.client.update_record(record_id, {"summarized_vacancy": new_summary})

    def create_vacancy(self, vacancy: str, summarized_vacancy: str = "") -> Optional[str]:
        """
        Create a new vacancy record in the Airtable.
        Returns the ID of the created record.
        """
        fields = {
            "vacancy": vacancy,
            "summarized_vacancy": summarized_vacancy,
        }
        response = self.client.create_record(fields)
        return response.get("id") if response else None
