import os
from typing import Optional

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException

from src.platform.cv_summarizer import CVSummarizer
from src.platform.dummy_predictor import DummyPredictor
from src.platform.lm_predictor import LMPredictor
from src.platform.vacancy_summarizer import VacancySummarizer
from src.service.airtable_client.airtable_client import AirtableClient
from src.service.airtable_client.cv_manager import CVManager
from src.service.airtable_client.pair_manager import PairManager
from src.service.airtable_client.vacancy_manager import VacancyManager
from src.service.models import (
    AvailableModelsPerPredictorResponse,
    AvailableModelsResponse,
    CreatePodResponse,
    MatchRequest,
    MatchResponse,
    PredictorParameters,
    PredictorType,
)
from src.service.runpod import RunPodManager

load_dotenv()

app = FastAPI(
    title="Candidate Scoring API",
    description="API for predicting candidate match scores for positions",
    version="1.0.0",
)

# Initialize summarizers with environment variable support
cv_summarizer = CVSummarizer(model="lmstudio-community/gemma-2-9b-it-GGUF")
vacancy_summarizer = VacancySummarizer(model="lmstudio-community/gemma-2-9b-it-GGUF")

# Initialize Airtable clients
cv_airtable_client = AirtableClient(
    api_key=os.getenv("AIRTABLE_API_KEY"),
    base_id=os.getenv("AIRTABLE_BASE_ID"),
    table_id=os.getenv("AIRTABLE_CV_TABLE_ID"),
)

vacancy_airtable_client = AirtableClient(
    api_key=os.getenv("AIRTABLE_API_KEY"),
    base_id=os.getenv("AIRTABLE_BASE_ID"),
    table_id=os.getenv("AIRTABLE_VACANCY_TABLE_ID"),
)

pair_airtable_client = AirtableClient(
    api_key=os.getenv("AIRTABLE_API_KEY"),
    base_id=os.getenv("AIRTABLE_BASE_ID"),
    table_id=os.getenv("AIRTABLE_PAIR_TABLE_ID"),
)

# Initialize managers
cv_manager = CVManager(airtable_client=cv_airtable_client)
vacancy_manager = VacancyManager(airtable_client=vacancy_airtable_client)
pair_manager = PairManager(airtable_client=pair_airtable_client)

# Initialize RunPod manager with error handling
try:
    runpod_manager = RunPodManager()
except Exception as e:
    print(f"Warning: Failed to initialize RunPod manager: {str(e)}")
    runpod_manager = None

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
        # Get the endpoint URL from environment or parameters
        api_base_url = (
            parameters.api_base_url  # type: ignore
            if parameters and parameters.api_base_url
            else os.getenv("RUNPOD_ENDPOINT_URL") or os.getenv("LM_API_BASE_URL")
        )

        if not api_base_url:
            raise HTTPException(
                status_code=400,
                detail="No LLM endpoint URL available. Please create a GPU pod first.",
            )

        return LMPredictor(
            api_base_url=api_base_url,
            api_key=parameters.api_key
            if parameters
            else os.getenv("LM_API_KEY", "not-needed"),  # type: ignore
            model=parameters.model
            if parameters
            else os.getenv("LM_MODEL", "QuantFactory/Meta-Llama-3-8B-GGUF"),  # type: ignore
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
    requested_vacancy = request.vacancy_description

    cv_id, cv_record, summarized_cv = cv_manager.find_cv(requested_cv)
    vacancy_id, vacancy_record, summarized_vacancy = vacancy_manager.find_vacancy(requested_vacancy)

    existing_pair = pair_manager.get_pair_result(cv_id, vacancy_id)
    if existing_pair:
        return MatchResponse(score=existing_pair["score"], description=existing_pair["comment"])
    
    if not cv_id:
        # If CV is not found, create a new record
        summarized_cv = cv_summarizer.summarize(requested_cv)
        cv_id = cv_manager.create_cv(requested_cv, summarized_cv)
    elif not summarized_cv:
        # If CV is found but not summarized, summarize it
        summarized_cv = cv_summarizer.summarize(cv_record)
        cv_manager.update_cv((cv_id, cv_record), summarized_cv)

    if not vacancy_id:
        # If vacancy is not found, create a new record
        summarized_vacancy = vacancy_summarizer.summarize(requested_vacancy)
        vacancy_id = vacancy_manager.create_vacancy(requested_vacancy, summarized_vacancy)
    elif not summarized_vacancy:
        # If vacancy is found but not summarized, summarize it
        summarized_vacancy = vacancy_summarizer.summarize(vacancy_record)
        vacancy_manager.update_vacancy((vacancy_id, vacancy_record), summarized_vacancy)

    score, comment = predictor.predict(
        summarized_cv, summarized_vacancy, request.hr_comment
    )
    pair_manager.save_pair_result(cv_id, vacancy_id, comment, score)

    return MatchResponse(score=score, description=comment)


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
    models_dict = {}

    for predictor_type, predictor_class in PREDICTOR_CLASSES.items():
        try:
            if predictor_type == "lm":
                # Initialize LMPredictor with current endpoint URL
                predictor = get_predictor(predictor_type)
            else:
                predictor = predictor_class()

            if predictor:
                models_dict[PredictorType(predictor_type)] = (
                    predictor.get_available_models()
                )
            else:
                models_dict[PredictorType(predictor_type)] = []

        except Exception as e:
            print(f"Warning: Failed to get models for {predictor_type}: {str(e)}")
            models_dict[PredictorType(predictor_type)] = []

    return AvailableModelsPerPredictorResponse(models=models_dict)


@app.post("/pods", response_model=CreatePodResponse)
async def create_pod():
    """Create a new RunPod instance."""
    if not runpod_manager:
        raise HTTPException(
            status_code=500,
            detail="RunPod manager not initialized. Check RUNPOD_API_KEY environment variable.",
        )

    result = runpod_manager.create_pod()
    if result:
        return CreatePodResponse(**result)
    raise HTTPException(status_code=500, detail="Failed to create pod")


@app.delete("/pods/{pod_id}")
async def delete_pod(pod_id: str):
    """Terminate a RunPod instance."""
    if not runpod_manager:
        raise HTTPException(status_code=500, detail="RunPod manager not initialized")

    try:
        runpod_manager.terminate_pod(pod_id)
        return {"status": "success"}
    except Exception as e:
        # Log the error but still return success if pod is not found
        print(f"Warning during pod termination: {str(e)}")
        return {"status": "success", "warning": "Pod may have already been terminated"}


@app.get("/pods")
async def list_pods():
    """List all RunPod instances."""
    if not runpod_manager:
        raise HTTPException(status_code=500, detail="RunPod manager not initialized")

    return {"pods": runpod_manager.list_pods()}


@app.get("/pods/{pod_id}/status")
async def check_pod_endpoint(pod_id: str):
    """Check if the pod's LLM endpoint is ready."""
    if not runpod_manager:
        raise HTTPException(status_code=500, detail="RunPod manager not initialized")

    return runpod_manager.check_endpoint_status(pod_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
