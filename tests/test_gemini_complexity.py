import pytest
import json
from unittest.mock import patch, MagicMock
from src.complexity.llm import LLMComplexityAnalyzer

@pytest.fixture
def gemini_analyzer():
    """Fixture to initialize GeminiComplexityAnalyzer."""
    performance_config_path = "tests/assests/model_preformance.json"
    return LLMComplexityAnalyzer(performance_config_path=performance_config_path)

@patch("src.complexity.gemini.completion")
def test_get_complexity_details(mock_completion, gemini_analyzer):
    """Test if GeminiComplexityAnalyzer returns valid complexity details."""

    # Mocking the API response from the Gemini LLM
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({
        "recommended_model": "openai/gpt-3.5-turbo",
        "overall_complexity": 75,
        "explanation": "This prompt requires strong reasoning skills, so GPT-3.5 was selected."
    })))]
    mock_completion.return_value = mock_response

    prompt = "Explain the theory of relativity in simple terms."

    details = gemini_analyzer.get_complexity_details(prompt)

    assert "recommended_model" in details
    assert "overall_complexity" in details
    assert "explanation" in details
    assert 0 <= details["overall_complexity"] <= 100
    assert details["recommended_model"] == "openai/gpt-3.5-turbo"

@patch("src.complexity.gemini.completion")
def test_calculate_complexity(mock_completion, gemini_analyzer):
    """Test if GeminiComplexityAnalyzer calculates a valid complexity score."""

    # Mock response for complexity score calculation
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({
        "recommended_model": "openai/gpt-4",
        "overall_complexity": 90.0,
        "explanation": "This is a highly technical prompt, requiring GPT-4."
    })))]
    mock_completion.return_value = mock_response

    prompt = "Derive the Schrödinger equation from first principles."

    score = gemini_analyzer.calculate_complexity(prompt)

    assert isinstance(score, (int, float))
    assert 0 <= score <= 100
    assert score == 90  # Mocked complexity score

@patch("src.complexity.gemini.completion")
def test_parse_json_response(mock_completion, gemini_analyzer):
    """Test if the JSON parsing method correctly extracts complexity details."""

    raw_json = json.dumps({
        "recommended_model": "gemini/gemini-1.5-flash",
        "overall_complexity": 60,
        "explanation": "This prompt balances reasoning and speed."
    })

    parsed_response = gemini_analyzer._parse_json_response(raw_json)

    assert parsed_response["recommended_model"] == "gemini/gemini-1.5-flash"
    assert parsed_response["overall_complexity"] == 60
    assert parsed_response["explanation"] == "This prompt balances reasoning and speed."

@patch("src.complexity.gemini.completion")
def test_handle_api_error(mock_completion, gemini_analyzer):
    """Test if the GeminiComplexityAnalyzer handles API errors gracefully."""

    mock_completion.side_effect = Exception("API error")

    prompt = "Write a Python script to sort a list of dictionaries by a key."

    details = gemini_analyzer.get_complexity_details(prompt)

    assert details["recommended_model"] == "Unknown"
    assert details["overall_complexity"] == -1
    assert "API error" in details["explanation"]
