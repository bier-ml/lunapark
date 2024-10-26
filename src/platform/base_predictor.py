from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BasePredictor(ABC):
    """Abstract base class for candidate-vacancy matching predictors."""

    @abstractmethod
    def predict(
        self, candidate_description: str, vacancy_description: str
    ) -> Tuple[float, Optional[str]]:
        """
        Predict match score between candidate and vacancy.

        Args:
            candidate_description (str): Description of the candidate's experience and skills
            vacancy_description (str): Description of the job vacancy requirements

        Returns:
            Tuple[float, str]: A tuple containing:
                - float: Match score between 0 and 1
                - str: Detailed description of the match analysis
        """
        pass

    @classmethod
    @abstractmethod
    def get_available_models(cls) -> Tuple[str, ...]:
        """Return tuple of available models for this predictor."""
        pass
