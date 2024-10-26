import os

import uvicorn
from fastapi import FastAPI, HTTPException

from src.platform.dummy_predictor import DummyPredictor
from src.platform.lm_predictor import LMPredictor
from src.service.models import (
    AvailableModelsResponse,
    MatchRequest,
    MatchResponse,
    PredictorType,
)

app = FastAPI(
    title="Candidate Scoring API",
    description="API for predicting candidate match scores for positions",
    version="1.0.0",
)

# Initialize predictors
predictors = {
    "dummy": DummyPredictor(),
    "lm": LMPredictor(
        api_base_url=os.getenv("LM_API_BASE_URL", "http://localhost:1234/v1"),
        api_key=os.getenv("LM_API_KEY", "not-needed"),
        model=os.getenv("LM_MODEL", "QuantFactory/Meta-Llama-3-8B-GGUF"),
    ),
}


@app.post(
    "/match",
    response_model=MatchResponse,
    summary="Calculate match score",
    description="Calculate a match score between a candidate and a position based on provided features",
)
async def calculate_match(request: MatchRequest) -> MatchResponse:
    """Calculate match score between vacancy and candidate."""
    predictor = predictors.get(request.predictor_type)
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
    available_types = [PredictorType(key) for key in predictors.keys()]
    return AvailableModelsResponse(predictor_types=available_types)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
