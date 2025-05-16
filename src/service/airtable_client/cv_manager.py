class CVManager:
    def __init__(self, airtable_client):
        self.client = airtable_client

    def retrieve_all_cvs(self):
        """
        Retrieve all CV records and return a dict:
        { record_id: (cv, summarized_cv) }
        """
        records = self.client.get_records()
        result = {}
        for record in records.get("records", []):
            record_id = record["id"]
            fields = record.get("fields", {})
            cv = fields.get("cv", "")
            summarized_cv = fields.get("summarized_cv", "")
            result[record_id] = (cv, summarized_cv)
        return result

    def find_unsummarized_cv(self, target_cv):
        """
        Find a record where the cv matches and summarized_cv is empty.
        Returns (record_id, cv) or None if not found.
        """
        cvs = self.retrieve_all_cvs()
        for record_id, (cv, summarized) in cvs.items():
            if cv == target_cv and not summarized:
                return record_id, cv
        return None

    def update_summarized_cv(self, record_id, new_summary):
        """
        Update a specific record with a new summarized_cv.
        """
        return self.client.update_record(record_id, {"summarized_cv": new_summary})
