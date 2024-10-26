"""
Test script for the candidate-position matching API.

This module provides functionality to test the prediction endpoint
of the matching service by sending sample requests and validating responses.
"""

import json
from typing import Dict, NoReturn

import requests
from fastapi.testclient import TestClient
from app import app  # Updated import

client = TestClient(app)


def test_prediction_endpoint() -> NoReturn:
    """
    Test the prediction endpoint of the matching API.

    Sends a POST request to the /predict endpoint with sample candidate
    and position data, then prints the response.

    Returns:
        None. Prints the API response or error message to console.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
    """

    # API endpoint URL
    url: str = "http://localhost:8000/predict"

    # Test data
    test_data: Dict[str, str] = {
        "candidate_features": "5 years of Python experience, Masters in Computer Science, Machine Learning and FastAPI expertise",
        "position_features": "3+ years Python experience required, Bachelor's degree in Computer Science, ML and FastAPI knowledge required",
    }

    try:
        # Make POST request to the API
        response: requests.Response = requests.post(
            url, json=test_data, headers={"Content-Type": "application/json"}
        )

        # Check if request was successful
        response.raise_for_status()

        # Print response
        print("\nAPI Response:")
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))

    except requests.exceptions.RequestException as e:
        print(f"\nError occurred: {e}")


def test_calculate_match():
    response = client.post(
        "/match",
        json={
            "vacancy_description": "Python developer with 3+ years of experience",
            "candidate_description": "5 years of Python development experience",
            "predictor_type": "dummy"
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert isinstance(data["score"], float)
    assert 0 <= data["score"] <= 1
    assert "description" in data
    assert isinstance(data["description"], str)


def test_invalid_predictor():
    response = client.post(
        "/match",
        json={
            "vacancy_description": "Python developer",
            "candidate_description": "Python experience",
            "predictor_type": "invalid_predictor"
        },
    )
    assert response.status_code == 400
    assert "Unsupported predictor type" in response.json()["detail"]


if __name__ == "__main__":
    print("Testing Candidate Scoring API...")
    test_prediction_endpoint()
