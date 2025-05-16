"""
Models for the candidate-position matching API.

This module defines the Pydantic models used for request and response validation
in the candidate-position matching service.
"""

import os
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from tortoise import fields
from tortoise.models import Model


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


class MatchResult(Model):
    """Database model for storing match results."""
    id = fields.IntField(pk=True)
    vacancy_description = fields.TextField()
    candidate_description = fields.TextField()
    hr_comment = fields.TextField()
    predictor_type = fields.CharField(max_length=50)
    score = fields.FloatField()
    description = fields.TextField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "match_results"

    def __str__(self):
        return f"MatchResult(id={self.id}, score={self.score}, created_at={self.created_at})"


class MatchService:
    """Service class for handling match result database operations."""
    
    @staticmethod
    async def save_match_result(
        vacancy_description: str,
        candidate_description: str,
        hr_comment: str,
        predictor_type: str,
        score: float,
        description: Optional[str] = None,
    ) -> MatchResult:
        """Save a match result to the database."""
        match_result = await MatchResult.create(
            vacancy_description=vacancy_description,
            candidate_description=candidate_description,
            hr_comment=hr_comment,
            predictor_type=predictor_type,
            score=score,
            description=description,
        )
        return match_result

    @staticmethod
    async def get_match_results(
        limit: int = 10,
        offset: int = 0,
    ) -> List[MatchResult]:
        """Get match results with pagination."""
        return await MatchResult.all().order_by("-created_at").offset(offset).limit(limit)

    @staticmethod
    async def get_match_result_by_id(match_id: int) -> Optional[MatchResult]:
        """Get a specific match result by ID."""
        return await MatchResult.get_or_none(id=match_id)

    @staticmethod
    async def find_existing_match(
        vacancy_description: str,
        candidate_description: str,
        predictor_type: str,
    ) -> Optional[MatchResult]:
        """
        Find an existing match result for the given vacancy and candidate descriptions.
        
        Args:
            vacancy_description: The job description
            candidate_description: The candidate's profile
            predictor_type: The type of predictor used
            
        Returns:
            Optional[MatchResult]: The existing match result if found, None otherwise
        """
        return await MatchResult.filter(
            vacancy_description=vacancy_description,
            candidate_description=candidate_description,
            predictor_type=predictor_type,
        ).order_by("-created_at").first()
