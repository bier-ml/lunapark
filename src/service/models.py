"""
Models for the candidate-position matching API.

This module defines the Pydantic models used for request and response validation
in the candidate-position matching service.
"""

import os
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictorType(str, Enum):
    DUMMY = "dummy"
    LM = "lm"
    # Add more predictor types here as they are implemented


class PredictorParameters(BaseModel):
    api_base_url: Optional[str] = Field(
        default=os.getenv("LM_API_BASE_URL", "http://localhost:5001/v1"),
        description="Base URL for the language model API",
    )
    api_key: Optional[str] = Field(
        default=None, description="API key for the language model service"
    )
    model: Optional[str] = Field(
        default=None, description="Model identifier to use for prediction"
    )


class MatchRequest(BaseModel):
    vacancy_description: str = Field(
        ...,
        description="The job description or requirements for the position",
        min_length=10,
    )
    candidate_description: str = Field(
        ...,
        description="The candidate's profile, experience, or resume text",
        min_length=10,
    )
    hr_comment: str = Field(
        ...,
        description="Any types of comments",
        min_length=0,
    )
    predictor_type: PredictorType = Field(
        default=PredictorType.DUMMY,
        description="The type of predictor to use for matching",
    )
    predictor_parameters: Optional[PredictorParameters] = Field(
        default=None, description="Optional parameters for the predictor configuration"
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


class AvailableModelsPerPredictorResponse(BaseModel):
    models: Dict[PredictorType, List[str]] = Field(
        description="Dictionary mapping predictor types to their available models"
    )
