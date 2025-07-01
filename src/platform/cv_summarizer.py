import requests
import os
from src.platform.base_summarizer import BaseSummarizer
import unicodedata

class CVSummarizer(BaseSummarizer):
    """
    A summarizer that uses Language Models via API (OpenAI-compatible) for CV summarization.
    """

    def __init__(
        self,
        api_base_url: str = None,
        api_key: str = "not-needed",  # LM Studio, for example, doesn't need real key
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Initialize the CV summarizer.

        Args:
            api_base_url: Base URL for the API endpoint
            api_key: API key for authentication
            model: Model identifier to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in the response
        """
        super().__init__()
        self.api_base_url = api_base_url or os.getenv("LM_API_BASE_URL", "http://host.docker.internal:5001/v1")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

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
            "messages": [{
                    "role": "system",
                    "content": """<|im_start|>system
You are an expert summarizer. Summarize the provided CV into the following five clearly defined categories:

1. Job Experience – List relevant work history, including roles, companies, durations, key responsibilities, and notable accomplishments.
2. Education – Include degrees, institutions, fields of study, and graduation years (if available).
3. Achievements – Highlight significant accomplishments such as awards, recognitions, or major milestones.
4. Publications – Mention any published works, articles, whitepapers, or speaking engagements.
5. Skills – Summarize technical and soft skills, including programming languages, tools, platforms, and relevant competencies.

Do not include any personal information such as names, addresses, phone numbers, emails, or links. Present the summary in a clean and readable format, using bullet points or concise text where appropriate.
<|im_end|>"""},
                {"role": "user", "content": self.clean_unicode_text(prompt)}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            f"{self.api_base_url}/chat/completions",
            headers=headers,
            json=payload,
        )

        if response.status_code != 200:
            raise Exception(f"API call failed with status code {response.status_code}")

        return response.json()["choices"][0]["message"]["content"]
    
    def summarize(self, text_to_summarize: str) -> str:
        """
        Summarize the given text.

        Args:
            text_to_summarize (str): Text to be summarized.

        Returns:
            str: Summarized text.
        """
        prompt = f"Summarize the following CV:\n{text_to_summarize}"
        return self._call_api(prompt)
    
    def get_available_models(self) -> tuple[str, ...]:
        """Return tuple of available models for this summarizer."""
        return ("lmstudio-community/gemma-2-9b-it-GGUF")
    
    def clean_unicode_text(self, text):
        # Keep only letters, numbers, punctuation, and whitespace
        return ''.join(
            ch for ch in text
            if unicodedata.category(ch).startswith(('L', 'N', 'P', 'Z'))
        )
