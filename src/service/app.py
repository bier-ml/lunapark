import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from src.platform.dummy_predictor import DummyPredictor
from src.platform.lm_predictor import LMPredictor
from src.service.models import (
    AvailableModelsPerPredictorResponse,
    AvailableModelsResponse,
    MatchRequest,
    MatchResponse,
    PredictorParameters,
    PredictorType,
)

app = FastAPI(
    title="Candidate Scoring API",
    description="API for predicting candidate match scores for positions",
    version="1.0.0",
)

PREDICTOR_CLASSES = {
    "dummy": DummyPredictor,
    "lm": LMPredictor,
}


def get_predictor(
    predictor_type: str, parameters: Optional[PredictorParameters] = None
):
    """Create a predictor instance with given parameters or default configuration."""
    if predictor_type == "dummy":
        return DummyPredictor()
    elif predictor_type == "lm":
        return LMPredictor(
            api_base_url=parameters.api_base_url  # type: ignore
            if parameters
            else os.getenv(
                "LM_API_BASE_URL", "http://localhost:1234/v1"
            ),  # base host for LMStudio
            api_key=parameters.api_key  # type: ignore
            if parameters
            else os.getenv("LM_API_KEY", "not-needed"),
            model=parameters.model  # type: ignore
            if parameters
            else os.getenv("LM_MODEL", "QuantFactory/Meta-Llama-3-8B-GGUF"),
        )
    return None


@app.post(
    "/match",
    response_model=MatchResponse,
    summary="Calculate match score",
    description="Calculate a match score between a candidate and a position based on provided features",
)
async def calculate_match(request: MatchRequest) -> MatchResponse:
    """Calculate match score between vacancy and candidate."""
    predictor = get_predictor(request.predictor_type, request.predictor_parameters)
    if not predictor:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported predictor type: {request.predictor_type}",
        )

    score, description = predictor.predict(
        request.candidate_description, request.vacancy_description
    )

    return MatchResponse(score=score, description=description)


@app.get(
    "/available-models",
    response_model=AvailableModelsResponse,
    summary="Get available predictor types",
    description="Returns a list of available predictor types that can be used for matching",
)
async def get_available_models() -> AvailableModelsResponse:
    """Get list of available predictor types."""
    available_types = [PredictorType(key) for key in PREDICTOR_CLASSES.keys()]
    return AvailableModelsResponse(predictor_types=available_types)


@app.get(
    "/available-models-per-predictor",
    response_model=AvailableModelsPerPredictorResponse,
    summary="Get available models for each predictor type",
    description="Returns a dictionary mapping predictor types to their available models",
)
async def get_available_models_per_predictor() -> AvailableModelsPerPredictorResponse:
    """Get available models for each predictor type."""
    models_dict = {
        PredictorType(predictor_type): predictor_class.get_available_models()  # type: ignore
        for predictor_type, predictor_class in PREDICTOR_CLASSES.items()
    }
    return AvailableModelsPerPredictorResponse(models=models_dict)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
