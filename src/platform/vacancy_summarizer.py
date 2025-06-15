import unicodedata
import requests
from src.platform.base_summarizer import BaseSummarizer


class VacancySummarizer(BaseSummarizer):
    """
    A summarizer that uses Language Models via API (OpenAI-compatible) for Vacancy summarization.
    """

    def __init__(
        self,
        api_base_url: str = "http://host.docker.internal:5001/v1",
        api_key: str = "not-needed",  # LM Studio, for example, doesn't need real key
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Initialize the Vacancy summarizer.

        Args:
            api_base_url: Base URL for the API endpoint
            api_key: API key for authentication
            model: Model identifier to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in the response
        """
        super().__init__()
        self.api_base_url = api_base_url
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
You are an expert summarizer. Summarize the provided job vacancy into the following five clearly defined categories:

1. Job Title and Level – Clearly state the title of the position and its seniority level (e.g., Junior, Senior, Lead).
2. Responsibilities – List the key duties, tasks, and expectations for the role.
3. Requirements – Summarize required qualifications, experience, skills, and educational background.
4. Preferred Qualifications – Include any optional or nice-to-have qualifications, experiences, or skills.
5. Technologies and Tools – Extract all mentioned technical tools, programming languages, frameworks, platforms, and methodologies.

Do not include any company-specific promotional text, location, compensation, or benefits unless they directly relate to the job’s technical or skill requirements. Present the summary in a clean and readable format, using bullet points or concise sections.
<|im_end|>"""},
                {"role": "user", "content": self.clean_unicode_text(prompt)}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
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
        prompt = f"Summarize the following vacancy:\n{text_to_summarize}"
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
