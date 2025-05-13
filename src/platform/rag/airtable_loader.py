import json
import os
from typing import Dict, List, Optional

import pandas as pd
from pyairtable import Api

from src.platform.graph.knowledge_graph import KnowledgeGraph
from src.platform.lm_predictor import LMPredictor
from src.platform.rag.resume_parser import ResumeParser


class AirtableLoader:
    """
    Load data from Airtable into Neo4j stores.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_id: Optional[str] = None,
        table_id: Optional[str] = None,
        lm_api_base_url: Optional[str] = None,
        lm_api_key: Optional[str] = None,
        lm_model: Optional[str] = None,
    ):
        """
        Initialize the Airtable loader.

        Args:
            api_key: Airtable API key
            base_id: Airtable base ID
            table_id: Airtable table ID
            lm_api_base_url: Base URL for LM API
            lm_api_key: API key for LM
            lm_model: Model identifier for LM
        """
        self.api_key = api_key or os.environ.get("AIRTABLE_TOKEN")
        self.base_id = base_id or os.environ.get("AIRTABLE_BASE")
        self.table_id = table_id or os.environ.get("AIRTABLE_TABLE")

        if not all([self.api_key, self.base_id, self.table_id]):
            raise ValueError(
                "Missing Airtable credentials. Set API key, base ID, and table ID."
            )

        self.api = Api(self.api_key)
        self.table = self.api.table(self.base_id, self.table_id)

        # Use environment variables or default values for Neo4j connection
        neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

        self.knowledge_graph = KnowledgeGraph(
            uri=neo4j_uri, user=neo4j_user, password=neo4j_password
        )

        # Set default LM API URL if not provided
        if lm_api_base_url is None:
            lm_api_base_url = os.environ.get(
                "LM_API_BASE_URL", "http://localhost:1234/v1"
            )

        # Ensure the URL has a scheme
        if lm_api_base_url and not lm_api_base_url.startswith(("http://", "https://")):
            lm_api_base_url = f"http://{lm_api_base_url}"

        # Initialize LM predictor for enhanced extraction
        self.lm_predictor = LMPredictor(
            api_base_url=lm_api_base_url,
            api_key=lm_api_key or os.environ.get("LM_API_KEY", "not-needed"),
            model=lm_model or os.environ.get("LM_MODEL", "local-model"),
        )

        # Initialize resume parser
        self.resume_parser = ResumeParser(
            lm_api_base_url=lm_api_base_url,
            lm_api_key=lm_api_key or os.environ.get("LM_API_KEY", "not-needed"),
            lm_model=lm_model or os.environ.get("LM_MODEL", "local-model"),
        )

    def get_all_records(self) -> List[Dict]:
        """Get all records from the Airtable table."""
        records = self.table.all()
        return [record["fields"] for record in records]

    def load_data_to_dataframe(self) -> pd.DataFrame:
        """Load all Airtable data into a pandas DataFrame."""
        records = self.get_all_records()
        return pd.DataFrame(records)

    def extract_candidate_info(self, data: pd.DataFrame) -> List[Dict]:
        """
        Extract candidate information from the DataFrame.
        Assumes specific column names from Airtable, adjust as needed.
        """
        candidates = []

        # Filter for candidate records and extract info
        candidate_data = data[
            ["Кандидат", "Текст резюме Rollup (from Кандидат)"]
        ].drop_duplicates()

        for _, row in candidate_data.iterrows():
            candidate = {
                "id": str(hash(row["Кандидат"])),  # Generate an ID from the name
                "name": row["Кандидат"],
                "resume_text": row["Текст резюме Rollup (from Кандидат)"],
            }
            candidates.append(candidate)

        return candidates

    def extract_vacancy_info(self, data: pd.DataFrame) -> List[Dict]:
        """
        Extract vacancy information from the DataFrame.
        Assumes specific column names from Airtable, adjust as needed.
        """
        vacancies = []

        # Check if the vacancy columns exist
        if (
            "Вакансия" in data.columns
            and "Текст вакансии Rollup (from Вакансия)" in data.columns
        ):
            # Filter for vacancy records and extract info
            vacancy_data = data[
                ["Вакансия", "Текст вакансии Rollup (from Вакансия)"]
            ].drop_duplicates()

            for _, row in vacancy_data.iterrows():
                if pd.notna(row["Вакансия"]) and pd.notna(
                    row["Текст вакансии Rollup (from Вакансия)"]
                ):
                    vacancy = {
                        "id": str(
                            hash(row["Вакансия"])
                        ),  # Generate an ID from the title
                        "title": row["Вакансия"],
                        "description": row["Текст вакансии Rollup (from Вакансия)"],
                    }
                    vacancies.append(vacancy)

        return vacancies

    def parse_skills_from_text(self, text: str) -> List[str]:
        """
        Extract skills from text using LLM.

        Args:
            text: Text to extract skills from

        Returns:
            List of extracted skills
        """
        prompt = f"""<|im_start|>system
You are an advanced AI assistant specialized in extracting technical skills and technologies from text.
Your task is to analyze the provided text and extract a list of skills in a structured JSON format.

You MUST return a valid JSON object with the following schema:
{{
    "skills": [string]  // List of technical skills, technologies, and tools mentioned
}}

Important guidelines:
1. Extract ONLY clearly mentioned technical skills, technologies, programming languages, frameworks, tools.
2. Return an empty list if no skills are found.
3. Do not include soft skills or generic terms unless they are clearly technical terms.
4. Do not make up skills that aren't mentioned in the text.
<|im_end|>

<|im_start|>user
Please extract all technical skills from the following text:

```
{text}
```
<|im_end|>"""

        try:
            # Call LLM API to extract skills
            response = self.lm_predictor._call_api(prompt)

            # Find the first opening brace and the last closing brace
            start_idx = response.find("{")
            end_idx = response.rfind("}")

            if start_idx == -1 or end_idx == -1:
                return []

            json_str = response[start_idx : end_idx + 1]

            # Parse the JSON
            result = json.loads(json_str)

            # Return the skills list
            return result.get("skills", [])

        except Exception as e:
            print(f"Error extracting skills: {str(e)}")
            return []

    def parse_location_from_text(self, text: str) -> Optional[str]:
        """
        Extract location from text using LLM.

        Args:
            text: Text to extract location from

        Returns:
            Extracted location or None if not found
        """
        prompt = f"""<|im_start|>system
You are an advanced AI assistant specialized in extracting location information from text.
Your task is to analyze the provided text and extract any mentioned city, country, or region in a structured JSON format.

You MUST return a valid JSON object with the following schema:
{{
    "location": string | null  // The most prominently mentioned location
}}

Important guidelines:
1. Extract ONLY clearly mentioned locations (cities, countries, regions).
2. Return null if no location is found.
3. If multiple locations are mentioned, return only the most prominent one (likely the current location).
4. Do not make up locations that aren't mentioned in the text.
<|im_end|>

<|im_start|>user
Please extract the location from the following text:

```
{text}
```
<|im_end|>"""

        try:
            # Call LLM API to extract location
            response = self.lm_predictor._call_api(prompt)

            # Find the first opening brace and the last closing brace
            start_idx = response.find("{")
            end_idx = response.rfind("}")

            if start_idx == -1 or end_idx == -1:
                return None

            json_str = response[start_idx : end_idx + 1]

            # Parse the JSON
            result = json.loads(json_str)

            # Return the location
            return result.get("location")

        except Exception as e:
            print(f"Error extracting location: {str(e)}")
            return None

    def parse_experience_from_text(self, text: str) -> List[Dict]:
        """
        Extract experience details from text using LLM.

        Args:
            text: Text to extract experience from

        Returns:
            List of experience dictionaries
        """
        prompt = f"""<|im_start|>system
You are an advanced AI assistant specialized in extracting work experience information from text.
Your task is to analyze the provided text and extract any mentioned work experience in a structured JSON format.

You MUST return a valid JSON object with the following schema:
{{
    "experience": [
        {{
            "role": string,            // Job title or role
            "company": string,         // Company name
            "years": number | null,    // Years of experience (numeric value only)
            "location": string | null, // Location of the job
            "start_date": string | null, // Start date of the job
            "end_date": string | null,   // End date of the job or "Present"
            "description": string | null // Job description
        }}
    ]
}}

Important guidelines:
1. Extract ALL work experiences mentioned in the text.
2. For fields not mentioned, use null.
3. For years, calculate the total years spent at the company based on dates if possible.
4. Return an empty array if no work experience is found.
<|im_end|>

<|im_start|>user
Please extract work experience information from the following text:

```
{text}
```
<|im_end|>"""

        try:
            # Call LLM API to extract experience
            response = self.lm_predictor._call_api(prompt)

            # Find the first opening brace and the last closing brace
            start_idx = response.find("{")
            end_idx = response.rfind("}")

            if start_idx == -1 or end_idx == -1:
                return []

            json_str = response[start_idx : end_idx + 1]

            # Parse the JSON
            result = json.loads(json_str)

            # Return the experience list
            return result.get("experience", [])

        except Exception as e:
            print(f"Error extracting experience: {str(e)}")
            return []

    def load_to_knowledge_graph(self) -> None:
        """Load all candidate and vacancy data into the knowledge graph."""
        df = self.load_data_to_dataframe()

        # Load candidates
        candidates = self.extract_candidate_info(df)
        for candidate in candidates:
            # Use ResumeParser for comprehensive parsing if resume text is available
            if candidate.get("resume_text"):
                try:
                    # Parse the resume using LLM
                    parsed_data = self.resume_parser.parse_resume(
                        candidate["resume_text"]
                    )

                    if parsed_data.get("status") == "success":
                        # Load data into knowledge graph
                        self.resume_parser.load_into_knowledge_graph(
                            parsed_resume=parsed_data,
                            knowledge_graph=self.knowledge_graph,
                            candidate_id=candidate["id"],
                        )
                        continue  # Skip the manual extraction below
                except Exception as e:
                    print(f"Error parsing resume for {candidate['name']}: {str(e)}")

            # Fallback to simpler extraction if resume parsing fails
            # Add candidate to graph
            self.knowledge_graph.add_candidate(
                candidate_id=candidate["id"], name=candidate["name"]
            )

            # Extract and add skills
            if candidate.get("resume_text"):
                skills = self.parse_skills_from_text(candidate["resume_text"])
                for skill in skills:
                    self.knowledge_graph.add_skill_to_candidate(
                        candidate_id=candidate["id"], skill_name=skill
                    )

                # Extract and add location
                location = self.parse_location_from_text(candidate["resume_text"])
                if location:
                    self.knowledge_graph.add_location_to_candidate(
                        candidate_id=candidate["id"], location=location
                    )

                # Extract and add experience
                experiences = self.parse_experience_from_text(candidate["resume_text"])
                for exp in experiences:
                    self.knowledge_graph.add_experience_to_candidate(
                        candidate_id=candidate["id"],
                        company=exp.get("company", "Unknown"),
                        role=exp.get("role", "Unknown"),
                        years=exp.get("years", 0) or 0,
                        metadata={
                            "start_date": exp.get("start_date"),
                            "end_date": exp.get("end_date"),
                            "location": exp.get("location"),
                            "description": exp.get("description"),
                        },
                    )

        # Load vacancies
        vacancies = self.extract_vacancy_info(df)
        for vacancy in vacancies:
            # Add vacancy to graph
            self.knowledge_graph.add_vacancy(
                vacancy_id=vacancy["id"], title=vacancy["title"]
            )

            if vacancy.get("description"):
                # Extract and add required skills
                skills = self.parse_skills_from_text(vacancy["description"])
                for skill in skills:
                    self.knowledge_graph.add_required_skill_to_vacancy(
                        vacancy_id=vacancy["id"], skill_name=skill
                    )

                # Extract and add location
                location = self.parse_location_from_text(vacancy["description"])
                if location:
                    self.knowledge_graph.add_location_to_vacancy(
                        vacancy_id=vacancy["id"], location=location
                    )

    def load_all_data(self) -> None:
        """Load all data to both the embeddings store and knowledge graph."""
        self.load_to_knowledge_graph()

    def close(self) -> None:
        """Close database connections."""
        self.knowledge_graph.close()
