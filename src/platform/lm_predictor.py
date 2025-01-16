import os
from typing import Optional, Tuple

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
        max_tokens: int = 4096,
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
                    "content": "You are a helpful assistant that evaluates job candidate matches. Always provide a score between 0 and 1, followed by a detailed explanation.",
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
        prompt = self.prompt_template.format(
            few_shot_examples=FEW_SHOT_EXAMPLES,
            vacancy_description=vacancy_description,
            candidate_description=candidate_description,
        )

        print("Generated prompt:", prompt)  # Log the generated prompt

        try:
            response = "<thought>\n" + self._call_api(prompt)
            print("API response received:", response)  # Log the API response

            # Extract the thought/analysis and score from the response
            thought = ""
            score = 0.0

            if "</thought>" in response:
                thought = response.split("<thought>")[1].split("</thought>")[0].strip()
                print("Extracted thought:", thought)  # Log the extracted thought

            if "<score>" in response and "</score>" in response:
                try:
                    score_str = (
                        response.split("<score>")[1].split("</score>")[0].strip()
                    )
                    score = float(score_str) / 100.0  # Convert 0-100 score to 0-1
                    score = max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
                    print("Extracted score:", score)  # Log the extracted score
                except (ValueError, IndexError):
                    print("Error parsing score, defaulting to 0.0")  # Log parsing error
                    score = 0.0

            return score, thought

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
