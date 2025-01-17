import random
from typing import Tuple

from src.platform.base_predictor import BasePredictor


class DummyPredictor(BasePredictor):
    """A simple dummy predictor that generates random scores and basic descriptions."""

    def predict(
        self,
        candidate_description: str,
        vacancy_description: str,
        hr_comment: str,
    ) -> tuple[float, str]:
        """
        Generate a random score and simple description for testing purposes.

        Args:
            candidate_description (str): Description of the candidate's experience and skills
            vacancy_description (str): Description of the job vacancy requirements
            hr_comment (str): HR comments about candidate's experience

        Returns:
            Tuple[float, str]: Random score between 0.1 and 0.9 and a basic description
        """

        _, _, _ = candidate_description, vacancy_description, hr_comment

        # Generate random score between 0.1 and 0.9
        score = round(random.uniform(0.1, 0.9), 2)

        # Generate basic description
        if score >= 0.7:
            description = "Strong match! The candidate appears to have most of the required skills."
        elif score >= 0.4:
            description = "Moderate match. Some skills align with the requirements."
        else:
            description = "Limited match. Few skills align with the job requirements."

        return score, description

    def get_available_models(self) -> Tuple[str]:
        return ("dummy-model-v1",)  # Dummy predictor has only one model
