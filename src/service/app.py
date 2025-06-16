import os
from typing import Optional

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException

from src.platform.cv_summarizer import CVSummarizer
from src.platform.dummy_predictor import DummyPredictor
from src.platform.lm_predictor import LMPredictor
from src.platform.rag.graph_rag_predictor import GraphRAGPredictor
from src.platform.vacancy_summarizer import VacancySummarizer
from src.service.airtable_client.airtable_client import AirtableClient
from src.service.airtable_client.cv_manager import CVManager
from src.service.airtable_client.pair_manager import PairManager
from src.service.airtable_client.vacancy_manager import VacancyManager
from src.service.models import (
    AvailableModelsPerPredictorResponse,
    AvailableModelsResponse,
    CandidateSearchRequest,
    CandidateSearchResponse,
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


# Initialize RunPod manager as a global variable, but with proper error handling
try:
    runpod_manager = RunPodManager()
except Exception as e:
    print(f"Warning: Failed to initialize RunPod manager: {str(e)}")
    runpod_manager = None

cv_summarizer = CVSummarizer(model="lmstudio-community/gemma-2-9b-it-GGUF")
vacancy_summarizer = VacancySummarizer(model="lmstudio-community/gemma-2-9b-it-GGUF")

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

cv_manager = CVManager(airtable_client=cv_airtable_client)
vacancy_manager = VacancyManager(airtable_client=vacancy_airtable_client)
pair_manager = PairManager(airtable_client=pair_airtable_client)

cv_summarizer = CVSummarizer(model="lmstudio-community/gemma-2-9b-it-GGUF")
vacancy_summarizer = VacancySummarizer(model="lmstudio-community/gemma-2-9b-it-GGUF")

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

cv_manager = CVManager(airtable_client=cv_airtable_client)
vacancy_manager = VacancyManager(airtable_client=vacancy_airtable_client)
pair_manager = PairManager(airtable_client=pair_airtable_client)


# Initialize RunPod manager as a global variable, but with proper error handling
try:
    runpod_manager = RunPodManager()
except Exception as e:
    print(f"Warning: Failed to initialize RunPod manager: {str(e)}")
    runpod_manager = None

# Initialize RunPod manager as a global variable, but with proper error handling
try:
    runpod_manager = RunPodManager()
except Exception as e:
    print(f"Warning: Failed to initialize RunPod manager: {str(e)}")
    runpod_manager = None

PREDICTOR_CLASSES = {
    "lm": LMPredictor,
    # "dummy": DummyPredictor,
    "graph_rag": GraphRAGPredictor,
}


def get_predictor(
    predictor_type: str, parameters: Optional[PredictorParameters] = None
):
    """Create a predictor instance with given parameters or default configuration."""
    # if predictor_type == "dummy":
    #     return DummyPredictor()
    if predictor_type == "lm":
        # Get the endpoint URL from parameters, or environment variables
        # For production: use RUNPOD_ENDPOINT_URL if available
        # For local development: use LM_API_BASE_URL with default http://localhost:1234/v1
        api_base_url = None
        if parameters and parameters.api_base_url:
            api_base_url = parameters.api_base_url
        elif os.getenv("RUNPOD_ENDPOINT_URL"):
            # If RUNPOD_ENDPOINT_URL is set, we're in production mode with RunPod
            api_base_url = os.getenv("RUNPOD_ENDPOINT_URL")
        else:
            # Default to local LLM
            api_base_url = os.getenv("LM_API_BASE_URL", "http://localhost:1234/v1")

        if not api_base_url:
            raise HTTPException(
                status_code=400,
                detail="No LLM endpoint URL available. Please create a GPU pod first.",
            )

        # Get temperature from parameters if provided
        temperature = None
        if parameters and parameters.temperature is not None:
            temperature = parameters.temperature
        else:
            temperature = float(os.getenv("LM_TEMPERATURE", "0.1"))

        # Get seed from parameters if provided
        seed = None
        if parameters and parameters.seed is not None:
            seed = parameters.seed
        else:
            seed = int(os.getenv("LM_SEED", "42")) if os.getenv("LM_SEED") else None

        return LMPredictor(
            api_base_url=api_base_url,
            api_key=parameters.api_key
            if parameters
            else os.getenv("LM_API_KEY", "not-needed"),  # type: ignore
            model=parameters.model
            if parameters
            else os.getenv("LM_MODEL", "QuantFactory/Meta-Llama-3-8B-GGUF"),  # type: ignore
            temperature=temperature,
            seed=seed,
        )
    elif predictor_type == "graph_rag":
        # Get database connection settings from environment variables
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

        # Get LLM settings (for enriching explanations)
        api_base_url = os.getenv("LM_API_BASE_URL", "http://localhost:1234/v1")
        api_key = os.getenv("LM_API_KEY", "not-needed")
        model = os.getenv("LM_MODEL", "QuantFactory/Meta-Llama-3-8B-GGUF")

        # Override with parameters if provided
        if parameters:
            if parameters.api_base_url:
                api_base_url = parameters.api_base_url
            if parameters.api_key:
                api_key = parameters.api_key
            if parameters.model:
                model = parameters.model

        return GraphRAGPredictor(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            lm_api_base_url=api_base_url,
            lm_api_key=api_key,
            lm_model=model,
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

@app.get("/debug/candidates")
async def debug_candidates(job_query: str = "Data Scientist with PyTorch, Python, SQL"):
    """Debug endpoint for candidate search functionality."""
    print(f"Debug endpoint called with job_query: {job_query}")
    try:
        # Initialize the predictor
        predictor = get_predictor("graph_rag")
        if not predictor:
            return {
                "status": "error",
                "message": "Failed to initialize GraphRAG predictor",
            }

        try:
            # Count candidates and sample a few
            with predictor.driver.session() as session:
                result = session.run("MATCH (c:Candidate) RETURN count(c) as count")
                candidate_count = result.single()["count"] if result.peek() else 0

                result = session.run(
                    "MATCH (c:Candidate) RETURN c.id as id, c.name as name LIMIT 3"
                )
                candidates = [
                    {"id": record["id"], "name": record["name"]} for record in result
                ]

                result = session.run("MATCH (s:Skill) RETURN count(s) as count")
                skills_count = result.single()["count"] if result.peek() else 0

                # 1. Vector search
                try:
                    vector_results = predictor.score_candidates_vector(
                        job_query, top_k=3
                    )
                except Exception as e:
                    vector_results = {"error": str(e)}

                # 2. Graph search
                try:
                    graph_results = predictor.score_candidates_graph(job_query, top_k=3)
                except Exception as e:
                    graph_results = {"error": str(e)}

                return {
                    "status": "success",
                    "db_connection": "ok",
                    "candidate_count": candidate_count,
                    "candidates_sample": candidates,
                    "skills_count": skills_count,
                    "query": job_query,
                    "extracted_components": predictor.extract_query_components(
                        job_query
                    ),
                    "vector_search_results": vector_results,
                    "graph_search_results": graph_results,
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Neo4j connection error: {str(e)}",
                "db_uri": predictor.neo4j_uri,
                "db_user": predictor.neo4j_user,
            }

    except Exception as e:
        return {"status": "error", "message": f"Debug error: {str(e)}"}


@app.post(
    "/search_candidates",
    response_model=CandidateSearchResponse,
    summary="Search for candidates matching a job description",
    description="Find candidates that match the given job description or query based on skills, experience, and other factors",
)
async def search_candidates(request: CandidateSearchRequest) -> CandidateSearchResponse:
    """Search for candidates matching a job description or query."""
    print(f"Searching for candidates with query: {request.job_query[:50]}...")
    try:
        # Only graph_rag predictor supports candidate search
        if request.predictor_type != "graph_rag":
            return CandidateSearchResponse(
                candidates=[],
                message=f"Unsupported predictor type: {request.predictor_type}. Only 'graph_rag' supports candidate search.",
            )

        # Initialize the predictor
        predictor = get_predictor(request.predictor_type, request.predictor_parameters)
        if not predictor:
            return CandidateSearchResponse(
                candidates=[],
                message=f"Failed to initialize predictor: {request.predictor_type}",
            )

        # Perform the search
        # First extract components from the query if in natural language format
        if (
            request.job_query and len(request.job_query) > 30
        ):  # Likely a full description
            components = predictor.extract_query_components(request.job_query)
            print(f"Extracted components: {components}")
        else:
            # Simple query, use as is
            components = {
                "role": request.job_query,
                "location": "",
                "years": 0.0,
                "skills": [],
            }
            print(f"Using simple query components: {components}")

        # Get candidates based on search parameters
        candidates = []

        if request.include_vector_search:
            print("Performing vector search...")
            # Perform vector similarity search
            try:
                vector_results = predictor.score_candidates_vector(
                    request.job_query, top_k=request.top_k
                )
                print(f"Vector search found {len(vector_results)} candidates")
                candidates.extend(vector_results)
            except Exception as e:
                print(f"Vector search failed: {str(e)}")
                vector_results = []

        if request.include_graph_search:
            print("Performing graph search...")
            # Perform graph-based search
            try:
                graph_results = predictor.score_candidates_graph(
                    request.job_query, top_k=request.top_k
                )
                print(f"Graph search found {len(graph_results)} candidates")
                # Add to candidates list
                if graph_results:
                    candidates.extend(graph_results)
            except Exception as e:
                print(f"Graph search failed: {str(e)}")
                graph_results = []

        print(f"Found a total of {len(candidates)} candidates from all search methods")

        # If we have candidates from both sources, combine scores
        final_candidates = []
        if (
            request.include_vector_search
            and request.include_graph_search
            and "vector_results" in locals()
            and "graph_results" in locals()
            and vector_results
            and graph_results
        ):
            print("Combining scores from vector and graph search...")
            combined_results = predictor.combine_scores(vector_results, graph_results)
            final_candidates = combined_results
        elif (
            request.include_vector_search
            and "vector_results" in locals()
            and vector_results
        ):
            # Only vector search was enabled or successful
            print("Using only vector search results...")
            combined_results = predictor.combine_scores(vector_results, [])
            final_candidates = combined_results
        elif (
            request.include_graph_search
            and "graph_results" in locals()
            and graph_results
        ):
            # Only graph search was enabled or successful
            print("Using only graph search results...")
            combined_results = predictor.combine_scores([], graph_results)
            final_candidates = combined_results
        else:
            print("No results found from enabled search methods")
            return CandidateSearchResponse(
                candidates=[],
                message="No matching candidates found with the selected search methods",
            )

        # Limit to top_k
        final_candidates = final_candidates[: request.top_k]
        print(f"Final candidates after filtering: {len(final_candidates)}")

        # For each candidate, generate an analysis with pros and cons
        for candidate in final_candidates:
            # Extract candidate details from Neo4j if available
            candidate_id = candidate.get("id")
            if candidate_id:
                try:
                    # Get the explanation with pros and cons for this candidate
                    explanation, pros_cons = predictor.generate_match_explanation(
                        candidate_id=candidate_id,
                        job_description=request.job_query,
                        combined_score=candidate.get("combined_score", 0.0),
                    )

                    # Add to the result
                    candidate["explanation"] = explanation
                    candidate["pros_cons"] = pros_cons

                    # Extract matched skills from graph matches if available
                    graph_matches = candidate.get("graph_matches", [])
                    matched_skills = []
                    for match in graph_matches:
                        if "skill" in match:
                            skill_name = match.get("skill", {}).get("name")
                            if skill_name and skill_name not in matched_skills:
                                matched_skills.append(skill_name)

                    candidate["matched_skills"] = matched_skills
                except Exception as e:
                    print(
                        f"Error generating explanation for candidate {candidate_id}: {str(e)}"
                    )
                    candidate["explanation"] = "Failed to generate explanation."
                    candidate["pros_cons"] = {"pros": [], "cons": []}
                    candidate["matched_skills"] = []

        return CandidateSearchResponse(
            candidates=final_candidates,
            message=f"Found {len(final_candidates)} matching candidates",
        )

    except Exception as e:
        print(f"Candidate search error: {str(e)}")
        return CandidateSearchResponse(
            candidates=[], message=f"Failed to search candidates: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
