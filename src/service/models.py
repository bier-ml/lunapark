"""
Models for the candidate-position matching API.

This module defines the Pydantic models used for request and response validation
in the candidate-position matching service.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PredictorType(str, Enum):
    DUMMY = "dummy"
    # Add more predictor types here as they are implemented
    

class MatchRequest(BaseModel):
    vacancy_description: str
    candidate_description: str
    predictor_type: PredictorType = PredictorType.DUMMY  # Default to dummy predictor


class MatchResponse(BaseModel):
    score: float
    description: Optional[str] = None
