from abc import ABC, abstractmethod
from typing import Tuple

class BaseSummarizer(ABC):
    """Abstract base class for candidate-vacancy matching summarizers."""

    @abstractmethod
    def summarize(
        self,
        text_to_summarize: str,
    ) -> str:
        """
        Summarize the given text.
        Args:
            text_to_summarize (str): Text to be summarized.
        Returns:
            str: Summarized text.
        """
        pass

    @abstractmethod
    def get_available_models(self) -> Tuple[str, ...]:
        """Return tuple of available models for this summarizer."""
        pass
