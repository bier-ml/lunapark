# CV-Vacancy Matcher

A system for matching job candidates with vacancies using AI-powered analysis. The project consists of a FastAPI backend service, a Streamlit frontend application, and an LLM-based matching system.

## Design Documentation

For detailed information about the system architecture, components, and technical decisions, please refer to our [Design Document](design_doc.md). This document outlines the core system design, architecture decisions, and implementation details for developers and technical stakeholders.

## Repository Structure

```
.
├── src/
│   ├── app/              # Streamlit frontend application
│   │   ├── streamlit_app.py
│   │   └── __init__.py
│   ├── service/          # Backend service
│   │   ├── app.py
│   │   ├── models.py
│   │   ├── test_api.py
│   │   └── __init__.py
│   ├── platform/         # Core platform functionality
│   │   ├── lm_predictor.py
│   │   ├── prompts/
│   │   ├── rag/           # Retrieval-Augmented Generation components
│   │   │   ├── airtable_loader.py 
│   │   │   └── graph_rag_predictor.py
│   │   ├── graph/         # Graph database components
│   │   │   └── knowledge_graph.py
│   │   └── ...
├── notebooks/           # Jupyter notebooks
├── misc/                # Demo and example files
├── poetry.lock
├── pyproject.toml
├── requirements.txt
├── setup.py             # Package installation setup
├── Dockerfile.fastapi
└── Dockerfile.streamlit
```

## Setup

This project uses Poetry for dependency management. Follow these steps to set up your development environment:

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository and install dependencies:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   poetry install
   ```

3. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

## Running the Application

### Demo

![Demo](misc/demo-lunapark.gif)

Watch the demo above to see the CV-Vacancy Matcher in action.

### Backend Service

1. Start the FastAPI backend server:
   ```bash
   poetry run uvicorn src.service.app:app --reload
   ```

   The API will be available at `http://localhost:8000`

### Frontend Application

1. Make sure the backend service is running
2. In a new terminal, start the Streamlit app:
   ```bash
   poetry run streamlit run src/app/streamlit_app.py
   ```

   The Streamlit interface will be available at `http://localhost:8501`

### Graph RAG Pipeline

The project includes a Graph RAG pipeline for more advanced matching between candidates and job vacancies.

#### Components
- **Neo4j**: Graph database for structured relationships between candidates, skills, locations, etc.
- **Semantic Matching**: Uses sentence transformers to calculate semantic similarity
- **Graph Matching**: Uses graph structures to find candidates based on skills, experience, location

#### Setup

1. **Neo4j Database**: We use a Neo4j container. You can start it with Docker Compose:
   ```bash
   docker-compose up neo4j
   ```
   Access Neo4j Browser at http://localhost:7474 (default credentials: neo4j/password)

2. **Load Data from Airtable**:
   ```bash
   # Set environment variables or use .env file
   export AIRTABLE_TOKEN=your_token
   export AIRTABLE_BASE=your_base_id
   export AIRTABLE_TABLE=your_table_id
   
   # Run the loader script
   python -m src.platform.rag.load_airtable_data
   ```

#### Usage

When making requests to the matching API, you can now select `graph_rag` as the predictor type:

```json
{
  "candidate_description": "...",
  "vacancy_description": "...",
  "hr_comment": "",
  "predictor_type": "graph_rag"
}
```

The Graph RAG system combines vector similarity with graph-based matching to provide more accurate and explainable results.

## Development

### Code Quality

We use several tools to maintain code quality:

- isort: Import sorting
- ruff: Linting
- mypy: Type checking
- poetry-export: Dependencies management

Run all pre-commit checks:
```bash
poetry run pre-commit run --all-files
```

Or run individual checks:
```bash
poetry run pre-commit run isort --all-files
poetry run pre-commit run ruff --all-files
poetry run pre-commit run mypy --all-files
poetry run pre-commit run poetry-export --all-files
```

## Testing

Run tests using pytest:
```bash
poetry run pytest
```

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

This means you are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- Attribution — You must give appropriate credit to the original authors, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

See the [LICENSE](LICENSE) file in the repository root for the full license text.
