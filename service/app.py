import uvicorn
from fastapi import FastAPI, HTTPException
from service.models import MatchRequest, MatchResponse
from platform.dummy_predictor import DummyPredictor

app = FastAPI(
    title="Candidate Scoring API",
    description="API for predicting candidate match scores for positions",
    version="1.0.0",
)

# Initialize predictors
predictors = {
    "dummy": DummyPredictor()
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
            detail=f"Unsupported predictor type: {request.predictor_type}"
        )
    
    score, description = predictor.predict(
        request.candidate_description,
        request.vacancy_description
    )
    
    return MatchResponse(score=score, description=description)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
