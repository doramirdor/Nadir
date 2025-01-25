import pytest
import json
from unittest.mock import patch, MagicMock

from src.complextiy.analyzer import ComplexityAnalyzer
from src.complextiy.gemini import GeminiComplexityAnalyzer


@pytest.fixture
def mock_gemini_model():
    """
    A pytest fixture that patches the Gemini model to return a mock response.
    This avoids actual network calls.
    """
    with patch("google.generativeai.configure") as mock_config, \
         patch("google.generativeai.GenerativeModel") as mock_model_cls:
        
        # Mock model instance
        mock_model_instance = MagicMock()
        mock_model_cls.return_value = mock_model_instance
        
        yield mock_model_instance


def test_init_no_api_key():
    """
    Test that initializing without an API key raises a ValueError.
    """
    with pytest.raises(ValueError, match="A Google Generative AI API key is required."):
        GeminiComplexityAnalyzer(api_key=None)  # no env var, so should raise


def test_calculate_complexity_valid_json(mock_gemini_model):
    """
    Test a successful complexity calculation with valid JSON returned.
    """
    # Setup the mock to return a valid JSON response
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "overall_complexity": 75,
        "explanation": "Text is moderately complex."
    })
    mock_gemini_model.generate_content.return_value = mock_response

    # Create analyzer with a dummy key
    analyzer = GeminiComplexityAnalyzer(api_key="dummy-key")

    # Call the method under test
    prompt = "A somewhat complex text about microservices and architecture."
    score = analyzer.calculate_complexity(prompt)

    # Assertions
    assert score == 75, "Expected overall_complexity of 75 from mock JSON"
    mock_gemini_model.generate_content.assert_called_once()


def test_calculate_complexity_invalid_json(mock_gemini_model):
    """
    Test behavior when the Gemini model returns invalid JSON.
    The analyzer should fallback to overall_complexity=0.0 and explanation=raw_text.
    """
    # Setup the mock to return invalid JSON
    mock_response = MagicMock()
    mock_response.text = "This is not valid JSON"
    mock_gemini_model.generate_content.return_value = mock_response

    analyzer = GeminiComplexityAnalyzer(api_key="dummy-key")

    prompt = "A complex text that triggers invalid JSON response."
    details = analyzer.get_complexity_details(prompt)

    assert details["overall_complexity"] == -1.0
    assert details["explanation"] == "This is not valid JSON"
    mock_gemini_model.generate_content.assert_called_once()


def test_calculate_complexity_api_error(mock_gemini_model):
    """
    Test that a RuntimeError is raised when the Gemini API call fails.
    """
    # Configure the mock to raise an exception
    mock_gemini_model.generate_content.side_effect = Exception("Simulated API failure")

    analyzer = GeminiComplexityAnalyzer(api_key="dummy-key")
    prompt = "Testing error scenario."

    with pytest.raises(RuntimeError, match="Gemini API error:"):
        analyzer.get_complexity_details(prompt)

    mock_gemini_model.generate_content.assert_called_once()
