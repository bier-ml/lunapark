import json
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.platform.base_predictor import BasePredictor
from src.platform.prompts.simple_prompt import FEW_SHOT_EXAMPLES, PROMPT


class LMPredictor(BasePredictor):
    """
    A predictor that uses Language Models via API (OpenAI-compatible) for predictions.
    """

    def __init__(
        self,
        api_base_url: str = "http://localhost:1234/v1",  # Default for local LM Studio
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

        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

    def predict(
        self, candidate_description: str, vacancy_description: str
    ) -> Tuple[float, Optional[str]]:
        """
        Predict match score and generate description for candidate-vacancy pair.

        Args:
            candidate_description (str): Description of the candidate's experience and skills
            vacancy_description (str): Description of the job vacancy requirements

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

        print(prompt)

        try:
            response = "<thought>\n" + self._call_api(prompt)

            print(response)
            # Extract the thought/analysis and score from the response
            thought = ""
            score = 0.0

            if "</thought>" in response:
                thought = response.split("<thought>")[1].split("</thought>")[0].strip()

            if "<score>" in response and "</score>" in response:
                try:
                    score_str = (
                        response.split("<score>")[1].split("</score>")[0].strip()
                    )
                    score = float(score_str) / 100.0  # Convert 0-100 score to 0-1
                    score = max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
                except (ValueError, IndexError):
                    score = 0.0

            return score, thought

        except Exception as e:
            return 0.0, f"Error in prediction: {str(e)}"

    def batch_predict(
        self, input_data_list: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """
        Make predictions for multiple inputs.

        Args:
            input_data_list: List of input data dictionaries

        Returns:
            List of prediction results
        """
        return [self.predict(input_data) for input_data in input_data_list]

    @classmethod
    def get_available_models(cls) -> Tuple[str]:
        return (
            "QuantFactory/Meta-Llama-3-8B-GGUF",
            "mistralai/Mistral-7B-v0.1",
            "meta-llama/Llama-2-7b-chat-hf",
        )  # Add your actual available models here
