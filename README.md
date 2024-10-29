# CV-Vacancy Matcher

A system for matching job candidates with vacancies using AI-powered analysis. The project consists of a FastAPI backend service and a Streamlit frontend application.

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
│   └── platform/         # Core platform functionality
│       ├── lm_predictor.py
│       ├── prompts/
│       └── ...
├── notebooks/           # Jupyter notebooks
├── misc/               # Demo and example files
├── poetry.lock
├── pyproject.toml
├── requirements.txt
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

[Add your license information here]
