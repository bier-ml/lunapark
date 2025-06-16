import json
from typing import Any, Dict, Optional

from src.platform.lm_predictor import LMPredictor


class ResumeParser:
    """
    Uses LLM to parse resumes and extract structured information.
    The extracted information can be used to populate the knowledge graph.
    """

    def __init__(
        self,
        lm_api_base_url: Optional[str] = None,
        lm_api_key: Optional[str] = None,
        lm_model: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """
        Initialize the resume parser.

        Args:
            lm_api_base_url: Base URL for LLM API
            lm_api_key: API key for LLM
            lm_model: Model identifier for LLM
            temperature: Temperature for LLM generation
        """
        # Initialize LM predictor for resume parsing
        self.lm_predictor = LMPredictor(
            api_base_url=lm_api_base_url,
            api_key=lm_api_key,
            model=lm_model,
            temperature=temperature,
            # Increase max tokens for parsing longer resumes
            max_tokens=1024,
        )

    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        Parse a resume using LLM and extract structured information.

        Args:
            resume_text: The text of the resume to parse

        Returns:
            Dictionary with structured resume information
        """
        # Create a prompt for the LLM to parse the resume
        prompt = self._create_extraction_prompt(resume_text)

        try:
            # Call LLM API
            response = self.lm_predictor._call_api(prompt)

            # Parse response to extract structured data
            return self._parse_extraction_response(response)
        except Exception as e:
            print(f"Error parsing resume: {str(e)}")
            return {"error": str(e), "status": "failed"}

    def _create_extraction_prompt(self, resume_text: str) -> str:
        """
        Create a prompt for the LLM to extract information from the resume.

        Args:
            resume_text: The text of the resume

        Returns:
            Formatted prompt for the LLM
        """
        return f"""<|im_start|>system
You are an advanced AI assistant specialized in parsing resumes and extracting structured information. 
Your task is to analyze the provided resume and extract key data points in a structured JSON format.

You MUST return a valid JSON object with the following schema:
{{
    "basic_info": {{
        "name": string,
        "email": string | null,
        "phone": string | null,
        "location": string | null,
        "linkedin": string | null,
        "github": string | null,
        "website": string | null
    }},
    "summary": string | null,
    "skills": [
        {{
            "name": string,
            "level": "beginner" | "intermediate" | "advanced" | "expert" | null
        }}
    ],
    "experience": [
        {{
            "company": string,
            "role": string,
            "years": number,
            "technologies": [string]
        }}
    ],
    "education": [
        {{
            "institution": string,
            "degree": string | null,
            "field": string | null
        }}
    ],
    "certifications": [
        {{
            "name": string,
            "issuer": string | null,
            "date": string | null
        }}
    ],
    "languages": [
        {{
            "name": string,
            "proficiency": string | null
        }}
    ]
}}

Important guidelines:
1. For each field, make a best effort to extract information but use null if not found.
2. For experience years, calculate the total years worked at each company.
3. For technologies in experience, extract specific technologies mentioned for each role.
4. Clean up the text by removing formatting artifacts.
5. Make sure to return valid JSON that can be parsed programmatically.
<|im_end|>

<|im_start|>user
Please parse the following resume and extract structured information according to the specified format:

```
{resume_text}
```
<|im_end|>"""

    def _parse_extraction_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract the structured data.

        Args:
            response: The LLM response containing the structured data

        Returns:
            Dictionary with structured resume information
        """
        try:
            # Extract JSON from the response
            response = response.strip()

            # Find the first opening brace and the last closing brace
            start_idx = response.find("{")
            end_idx = response.rfind("}")

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No valid JSON found in the response")

            json_str = response[start_idx : end_idx + 1]

            # Parse the JSON
            result = json.loads(json_str)

            # Add status field
            result["status"] = "success"

            return result
        except Exception as e:
            print(f"Error parsing extraction response: {str(e)}")
            print(f"Response was: {response}")
            return {"error": str(e), "raw_response": response, "status": "failed"}

    def load_into_knowledge_graph(
        self,
        parsed_resume: Dict[str, Any],
        knowledge_graph,
        candidate_id: Optional[str] = None,
    ) -> str:
        """
        Load parsed resume data into a knowledge graph.

        Args:
            parsed_resume: Parsed resume data from parse_resume method
            knowledge_graph: Instance of KnowledgeGraph class
            candidate_id: Optional candidate ID (generated if not provided)

        Returns:
            The candidate ID
        """
        if parsed_resume.get("status") != "success":
            raise ValueError("Cannot load failed parse result into knowledge graph")

        # Generate candidate ID if not provided
        if not candidate_id:
            basic_info = parsed_resume.get("basic_info", {})
            name = basic_info.get("name", "unknown")
            candidate_id = f"candidate_{hash(name)}_{hash(str(basic_info))}"

        # Add candidate to knowledge graph
        knowledge_graph.add_candidate(
            candidate_id=candidate_id,
            name=parsed_resume.get("basic_info", {}).get("name", "Unknown"),
            metadata={
                "email": parsed_resume.get("basic_info", {}).get("email"),
                "phone": parsed_resume.get("basic_info", {}).get("phone"),
                "linkedin": parsed_resume.get("basic_info", {}).get("linkedin"),
                "github": parsed_resume.get("basic_info", {}).get("github"),
                "website": parsed_resume.get("basic_info", {}).get("website"),
                "summary": parsed_resume.get("summary"),
            },
        )

        # Add location if available
        location = parsed_resume.get("basic_info", {}).get("location")
        if location:
            knowledge_graph.add_location_to_candidate(
                candidate_id=candidate_id, location=location, is_current=True
            )

        # Add skills
        for skill in parsed_resume.get("skills", []):
            knowledge_graph.add_skill_to_candidate(
                candidate_id=candidate_id,
                skill_name=skill.get("name"),
                level=skill.get("level"),
            )

        # Add experience
        for exp in parsed_resume.get("experience", []):
            # Add basic experience
            knowledge_graph.add_experience_to_candidate(
                candidate_id=candidate_id,
                company=exp.get("company", "Unknown Company"),
                role=exp.get("role", "Unknown Role"),
                years=exp.get("years", 0),
                metadata={},  # Simplified metadata with no additional fields
            )

            # Add technologies as skills associated with this experience
            for tech in exp.get("technologies", []):
                knowledge_graph.add_skill_to_candidate(
                    candidate_id=candidate_id,
                    skill_name=tech,
                    level=None,  # Level is not specified for technologies
                )

        return candidate_id
