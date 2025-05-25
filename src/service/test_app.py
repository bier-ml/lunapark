import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from src.service.app import calculate_match
from src.service.models import MatchRequest, MatchResponse

@pytest.mark.asyncio
@patch("src.service.app.cv_manager")
@patch("src.service.app.cv_summarizer")
@patch("src.service.app.vacancy_manager")
@patch("src.service.app.vacancy_summarizer")
@patch("src.service.app.get_predictor")
async def test_calculate_match_creates_new_cv_and_vacancy(
    mock_get_predictor,
    mock_vacancy_summarizer,
    mock_vacancy_manager,
    mock_cv_summarizer,
    mock_cv_manager,
):
    # Arrange
    request = MatchRequest(
        predictor_type="dummy",
        predictor_parameters=None,
        candidate_description="John Doe, Python Developer, very skilled in Python and Django",
        vacancy_description="Looking for a Python Developer, middle level, with experience in Django",
        hr_comment=""
    )

    # CV not found
    mock_cv_manager.find_cv.return_value = (None, None, None)
    mock_cv_summarizer.summarize.return_value = "Summarized CV"
    mock_cv_manager.create_cv.return_value = "cv_id"

    # Vacancy not found
    mock_vacancy_manager.find_vacancy.return_value = (None, None, None)
    mock_vacancy_summarizer.summarize.return_value = "Summarized Vacancy"
    mock_vacancy_manager.create_vacancy.return_value = "vacancy_id"

    # Predictor
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = (0.85, "Match description")
    mock_get_predictor.return_value = mock_predictor

    # Act
    response = await calculate_match(request)

    # Assert
    mock_cv_summarizer.summarize.assert_called_once_with(request.candidate_description)
    mock_cv_manager.create_cv.assert_called_once_with(request.candidate_description, "Summarized CV")
    mock_vacancy_summarizer.summarize.assert_called_once_with(request.vacancy_description)
    mock_vacancy_manager.create_vacancy.assert_called_once_with(request.vacancy_description, "Summarized Vacancy")
    mock_predictor.predict.assert_called_once_with("Summarized CV", "Summarized Vacancy", "")
    assert isinstance(response, MatchResponse)
    assert response.score == 0.85
    assert response.description == "Match description"

@pytest.mark.asyncio
@patch("src.service.app.cv_manager")
@patch("src.service.app.cv_summarizer")
@patch("src.service.app.vacancy_manager")
@patch("src.service.app.vacancy_summarizer")
@patch("src.service.app.get_predictor")
async def test_calculate_match_existing_cv_and_vacancy(
    mock_get_predictor,
    mock_vacancy_summarizer,
    mock_vacancy_manager,
    mock_cv_summarizer,
    mock_cv_manager,
):
    # Arrange
    request = MatchRequest(
        predictor_type="dummy",
        predictor_parameters=None,
        candidate_description="Jane Doe, Data Scientist, experienced in machine learning and data analysis",
        vacancy_description="Looking for a Data Scientist, senior level, with expertise in machine learning",
        hr_comment="Excellent"
    )

    # CV found and summarized
    mock_cv_manager.find_cv.return_value = ("cv_id", "cv_record", "Summarized CV")
    # Vacancy found and summarized
    mock_vacancy_manager.find_vacancy.return_value = ("vacancy_id", "vacancy_record", "Summarized Vacancy")

    # Predictor
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = (0.95, "Great match")
    mock_get_predictor.return_value = mock_predictor

    # Act
    response = await calculate_match(request)

    # Assert
    mock_cv_summarizer.summarize.assert_not_called()
    mock_cv_manager.create_cv.assert_not_called()
    mock_vacancy_summarizer.summarize.assert_not_called()
    mock_vacancy_manager.create_vacancy.assert_not_called()
    mock_predictor.predict.assert_called_once_with("Summarized CV", "Summarized Vacancy", "Excellent")
    assert isinstance(response, MatchResponse)
    assert response.score == 0.95
    assert response.description == "Great match"

@pytest.mark.asyncio
@patch("src.service.app.cv_manager")
@patch("src.service.app.cv_summarizer")
@patch("src.service.app.vacancy_manager")
@patch("src.service.app.vacancy_summarizer")
@patch("src.service.app.get_predictor")
async def test_calculate_match_summarizes_if_not_summarized(
    mock_get_predictor,
    mock_vacancy_summarizer,
    mock_vacancy_manager,
    mock_cv_summarizer,
    mock_cv_manager,
):
    # Arrange
    request = MatchRequest(
        predictor_type="dummy",
        predictor_parameters=None,
        candidate_description="CV text for summarization test - John Smith, Software Engineer",
        vacancy_description="Vacancy text for summarization test - Senior Software Engineer",
        hr_comment="Comment"
    )

    # CV found but not summarized
    mock_cv_manager.find_cv.return_value = ("cv_id", "cv_record", None)
    mock_cv_summarizer.summarize.return_value = "Summarized CV"
    # Vacancy found but not summarized
    mock_vacancy_manager.find_vacancy.return_value = ("vacancy_id", "vacancy_record", None)
    mock_vacancy_summarizer.summarize.return_value = "Summarized Vacancy"

    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = (0.5, "Partial match")
    mock_get_predictor.return_value = mock_predictor

    # Act
    response = await calculate_match(request)

    # Assert
    mock_cv_summarizer.summarize.assert_called_once_with("cv_record")
    mock_cv_manager.update_cv.assert_called_once_with(("cv_id", "cv_record"), "Summarized CV")
    mock_vacancy_summarizer.summarize.assert_called_once_with("vacancy_record")
    mock_vacancy_manager.update_vacancy.assert_called_once_with(("vacancy_id", "vacancy_record"), "Summarized Vacancy")
    mock_predictor.predict.assert_called_once_with("Summarized CV", "Summarized Vacancy", "Comment")
    assert isinstance(response, MatchResponse)
    assert response.score == 0.5
    assert response.description == "Partial match"