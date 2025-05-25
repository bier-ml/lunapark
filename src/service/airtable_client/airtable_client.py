import requests
from typing import Any, Dict, Optional

class AirtableClient:
    def __init__(self, api_key: str, base_id: str, table_id: str) -> None:
        self.api_key = api_key
        self.base_id = base_id
        self.table_id = table_id
        self.base_url = f"https://api.airtable.com/v0/{self.base_id}/{self.table_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_records(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fetch records from the Airtable table."""
        response = requests.get(self.base_url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def update_record(self, record_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a record in the Airtable table.

        :param record_id: The ID of the record to update.
        :param fields: A dictionary of fields to update.
        """
        url = f"{self.base_url}/{record_id}"
        data = {"fields": fields}
        response = requests.patch(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

    def create_record(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new record in the Airtable table.

        :param fields: A dictionary of fields for the new record.
        """
        data = {"fields": fields}
        response = requests.post(self.base_url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()
