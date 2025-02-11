import os
import pytest
import json
from unittest.mock import patch, MagicMock
from nadir.complexity.llm import LLMComplexityAnalyzer

@pytest.fixture
def llm_analyzer():
    """Fixture to initialize LLMComplexityAnalyzer."""
    return LLMComplexityAnalyzer(performance_config_path=os.path.join(os.path.dirname(__file__),"assets/model_performance.json"))


@patch("nadir.complexity.llm.completion")
def test_get_complexity_details(mock_completion, llm_analyzer):
    """Test if LLMComplexityAnalyzer returns valid complexity details."""

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = json.dumps({
        "recommended_model": "openai/gpt-3.5-turbo",
        "overall_complexity": 75,
        "explanation": "This prompt requires strong reasoning skills."
    })

    mock_completion.return_value = mock_response

    prompt = "Explain the theory of relativity."
    details = llm_analyzer.get_complexity_details(prompt)

    assert isinstance(details, dict)
    assert "recommended_model" in details
    assert "overall_complexity" in details
    assert "explanation" in details
    assert 0 <= details["overall_complexity"] <= 100
    assert details["recommended_model"] == "openai/gpt-3.5-turbo"

@patch("nadir.complexity.llm.completion")
def test_calculate_complexity(mock_completion, llm_analyzer):
    """Test if LLMComplexityAnalyzer calculates a valid complexity score."""

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = json.dumps({
        "recommended_model": "openai/gpt-4",
        "overall_complexity": 90,
        "explanation": "Highly technical prompt requiring GPT-4."
    })

    mock_completion.return_value = mock_response

    prompt = "Derive the Schrödinger equation."
    complexity_score = llm_analyzer.calculate_complexity(prompt)

    assert isinstance(complexity_score, (int, float))
    assert 0 <= complexity_score <= 100
    assert complexity_score == 90 

def test_parse_json_response(llm_analyzer):
    """Test if JSON response parsing correctly extracts complexity details."""

    raw_json = json.dumps({
        "recommended_model": "gemini/gemini-1.5-flash",
        "overall_complexity": 60,
        "explanation": "Balanced prompt complexity."
    })

    parsed_response = llm_analyzer._parse_json_response(raw_json)

    assert isinstance(parsed_response, dict)
    assert parsed_response["recommended_model"] == "gemini/gemini-1.5-flash"
    assert parsed_response["overall_complexity"] == 60
    assert parsed_response["explanation"] == "Balanced prompt complexity."
