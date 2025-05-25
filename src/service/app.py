import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from src.platform.cv_summarizer import CVSummarizer
from src.platform.dummy_predictor import DummyPredictor
from src.platform.lm_predictor import LMPredictor
from src.platform.vacancy_summarizer import VacancySummarizer
from src.service.airtable_client.airtable_client import AirtableClient
from src.service.airtable_client.cv_manager import CVManager
from src.service.airtable_client.vacancy_manager import VacancyManager
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

cv_summarizer = CVSummarizer(model="lmstudio-community/gemma-2-9b-it-GGUF")
vacancy_summarizer = VacancySummarizer(model="lmstudio-community/gemma-2-9b-it-GGUF")

cv_airtable_client = AirtableClient(
    api_key=os.getenv("AIRTABLE_API_KEY"),
    base_id="appPa8VJ4IHfm1V5O",
    table_id="tblF1QERP6FFNMnM1",
)

vacancy_airtable_client = AirtableClient(
    api_key=os.getenv("AIRTABLE_API_KEY"),
    base_id="appPa8VJ4IHfm1V5O",
    table_id="tblfVZLqyJjb2SVHW",
)

cv_manager = CVManager(airtable_client=cv_airtable_client)
vacancy_manager = VacancyManager(airtable_client=vacancy_airtable_client)

PREDICTOR_CLASSES = {
    "lm": LMPredictor,
    "dummy": DummyPredictor,
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
                "LM_API_BASE_URL", "http://localhost:5001/v1"
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
    
    requested_cv = request.candidate_description
    cv_id, cv_record, summarized_cv = cv_manager.find_cv(requested_cv)
    if not cv_id:
        # If CV is not found, create a new record
        summarized_cv = cv_summarizer.summarize(requested_cv)
        cv_id = cv_manager.create_cv(requested_cv, summarized_cv)
    elif not summarized_cv:
        # If CV is found but not summarized, summarize it
        summarized_cv = cv_summarizer.summarize(cv_record)
        cv_manager.update_cv((cv_id, cv_record), summarized_cv)

    requested_vacancy = request.vacancy_description
    vacancy_id, vacancy_record, summarized_vacancy = vacancy_manager.find_vacancy(requested_vacancy)
    if not vacancy_id:
        # If vacancy is not found, create a new record
        summarized_vacancy = vacancy_summarizer.summarize(requested_vacancy)
        vacancy_id = vacancy_manager.create_vacancy(requested_vacancy, summarized_vacancy)
    elif not summarized_vacancy:
        # If vacancy is found but not summarized, summarize it
        summarized_vacancy = vacancy_summarizer.summarize(vacancy_record)
        vacancy_manager.update_vacancy((vacancy_id, vacancy_record), summarized_vacancy)

    print(f"Summarized CV: {summarized_cv}")
    print("==" * 20)
    print(f"Summarized Vacancy: {summarized_vacancy}")
    score, description = predictor.predict(
        summarized_cv, summarized_vacancy, request.hr_comment
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
        PredictorType(predictor_type): predictor_class().get_available_models()  # type: ignore
        for predictor_type, predictor_class in PREDICTOR_CLASSES.items()
    }
    return AvailableModelsPerPredictorResponse(models=models_dict)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
