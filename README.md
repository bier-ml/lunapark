# Project Name

This repository contains a FastAPI backend service and a Streamlit frontend application.

## Repository Structure

```
.
├── app/                    # Streamlit frontend application
│   ├── streamlit_app.py   # Main Streamlit interface
│   ├── models.py          # Shared data models
│   ├── app.py            # Frontend application logic
│   └── test_api.py       # Frontend tests
├── service/              # Backend service
│   └── test_api.py      # Backend API tests
├── poetry.lock          # Poetry dependency lock file
├── pyproject.toml       # Project configuration and dependencies
└── requirements-service.txt  # Service-specific requirements
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

### Backend Service

1. Start the FastAPI backend server:
   ```bash
   poetry run uvicorn service.api:app --reload
   ```

   The API will be available at `http://localhost:8000`

### Frontend Application

1. Make sure the backend service is running
2. In a new terminal, start the Streamlit app:
   ```bash
   poetry run streamlit run app/streamlit_app.py
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
