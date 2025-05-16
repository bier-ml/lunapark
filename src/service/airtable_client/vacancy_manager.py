class VacancyManager:
    def __init__(self, airtable_client):
        self.client = airtable_client

    def retrieve_all_vacancies(self):
        """
        Retrieve all vacancy records and return a dict:
        { record_id: (vacancy, summarized_vacancy) }
        """
        records = self.client.get_records()
        result = {}
        for record in records.get("records", []):
            record_id = record["id"]
            fields = record.get("fields", {})
            vacancy = fields.get("vacancy", "")
            summarized_vacancy = fields.get("summarized_vacancy", "")
            result[record_id] = (vacancy, summarized_vacancy)
        return result

    def find_unsummarized_vacancy(self, target_vacancy):
        """
        Find a record where the vacancy matches and summarized_vacancy is empty.
        Returns (record_id, vacancy) or None if not found.
        """
        vacancies = self.retrieve_all_vacancies()
        for record_id, (vacancy, summarized) in vacancies.items():
            if vacancy == target_vacancy and not summarized:
                return record_id, vacancy
        return None

    def update_summarized_vacancy(self, record_id, new_summary):
        """
        Update a specific record with a new summarized_vacancy.
        """
        return self.client.update_record(record_id, {"summarized_vacancy": new_summary})
