"""
Models for the candidate-position matching API.

This module defines the Pydantic models used for request and response validation
in the candidate-position matching service.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictorType(str, Enum):
    DUMMY = "dummy"
    LM = "lm"
    # Add more predictor types here as they are implemented


class MatchRequest(BaseModel):
    vacancy_description: str = Field(
        ...,  # ... means the field is required
        description="The job description or requirements for the position",
        min_length=10,  # Ensure some minimal content
    )
    candidate_description: str = Field(
        ...,
        description="The candidate's profile, experience, or resume text",
        min_length=10,
    )
    predictor_type: PredictorType = Field(
        default=PredictorType.DUMMY,
        description="The type of predictor to use for matching",
    )


class MatchResponse(BaseModel):
    score: float = Field(
        ...,
        description="Matching score between 0 and 100",
        ge=0.0,  # greater than or equal to 0
        le=100.0,  # less than or equal to 100
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional explanation of the matching result",
    )


class AvailableModelsResponse(BaseModel):
    predictor_types: List[PredictorType] = Field(
        description="List of available predictor types that can be used for matching"
    )
