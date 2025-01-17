import os
from typing import Optional, Tuple
import re
import requests

from src.platform.base_predictor import BasePredictor
from src.platform.prompts.simple_prompt import FEW_SHOT_EXAMPLES, PROMPT




class LMPredictor(BasePredictor):
    """
    A predictor that uses Language Models via API (OpenAI-compatible) for predictions.
    """

    def __init__(
        self,
        api_base_url: str = os.getenv("LM_API_BASE_URL", "http://localhost:5001/v1"),
        api_key: str = "not-needed",  # LM Studio, for example, doesn't need real key
        model: str = "local-model",
        temperature: float = 0.7,
        max_tokens: int = 256,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize the LM predictor.

        Args:
            api_base_url: Base URL for the API endpoint
            api_key: API key for authentication
            model: Model identifier to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in the response
            prompt_template: Custom prompt template (uses default if None)
        """
        super().__init__()
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template or PROMPT

    def _call_api(self, prompt: str) -> str:
        """
        Make an API call to the language model.

        Args:
            prompt: The formatted prompt to send

        Returns:
            The model's response text

        Raises:
            Exception: If the API call fails
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": """<|im_start|>system
You are an advanced AI model designed to analyze the compatibility between a CV and a job description. You will receive a CV and a job description. Your task is to output a structured message in XML format that includes the following:
            
            1. though: Provide a short comment explaining your score.
            2. score: Provide a numerical compatibility score (1-5) based on qualifications, skills, and experience.

            Your output must be in XML format as follows:
            <thought> Your comment here. </thought>
            <score> Your score from 1 to 5 here </score>
<|im_end|>""",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Add detailed logging
        # print(f"Making API call to: {self.api_base_url}/chat/completions")
        # print(f"Request Headers: {headers}")
        # print(f"Request Payload: {payload}")

        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=180,
            )

            # Add response logging
            # print(f"Response Status Code: {response.status_code}")
            # print(f"Response Headers: {dict(response.headers)}")
            # print(f"Response Content: {response.text}")

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            # print(f"Request Exception: {str(e)}")
            if hasattr(e.response, "text"):
                print(f"Error Response Content: {e.response.text}")
            raise Exception(f"API call failed: {str(e)}")
        except Exception as e:
            # print(f"Unexpected error: {str(e, e.response.text)}")
            raise Exception(f"API call failed: {str(e, e.response.text)}")
    def parse_response(self, response: str):
        """Parse model response to extract thought and score"""
        # Try XML format first
        thought_match = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL)
        if thought_match is None:
            thought_match = re.search(r"<thoughts>(.*?)</thoughts>", response, re.DOTALL)
            
        score_start = response.find("<score>")
        score_end = response.find("</score>")
        if score_start != -1 and score_end != -1:
            score_content = response[score_start + len("<score>"):score_end].strip()
            if "/" in score_content:
                score_value = float(score_content.split("/")[0].strip())
            else:
                score_value = float(score_content)
        else:
            score_value = None
            
        if thought_match and score_value:
            thought = thought_match.group(1).strip()
            score = score_value
        else:
            # Fall back to plain text format
            score_match = re.search(r"(\d+\.?\d*)/5", response)
            score = float(score_match.group(1)) if score_match else None
            
            # Remove score from thought if present
            thought = re.sub(r"\d+\.?\d*/5", "", response).strip()
        return {"thought": thought, "score": score}
    def predict(
        self,
        candidate_description: str,
        vacancy_description: str,
        hr_comment: str,
    ) -> Tuple[float, Optional[str]]:
        """
        Predict match score and generate description for candidate-vacancy pair.

        Args:
            candidate_description (str): Description of the candidate's experience and skills
            vacancy_description (str): Description of the job vacancy requirements
            hr_comment (str): HR comments about candidate's experience

        Returns:
            Tuple[float, str]: A tuple containing:
                - float: Match score between 0 and 1
                - str: Detailed description of the match analysis
        """
        # Use the predefined prompt template
        prompt = f"""<|im_start|>user
                Please evaluate this candidate:
                <CV> 
                {candidate_description} 
                </CV>
                <job_description> 
                {vacancy_description} 
                </job_description>
                <|im_end|>""".format(
            vacancy_description=vacancy_description,
            candidate_description=candidate_description,
        )

        try:
            response = self._call_api(prompt)
            result = self.parse_response(response)
            return result['score'], result['thought']

        except Exception as e:
            print("Error during prediction:", str(e))  # Log the error
            return 0.0, f"Error in prediction: {str(e)}"

    def get_available_models(self) -> Tuple[str, ...]:
        try:
            response = requests.get(
                f"{self.api_base_url}/models",
            )
            response.raise_for_status()
            return [model["id"] for model in response.json()["data"]]
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
