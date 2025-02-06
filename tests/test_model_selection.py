import pytest
from unittest.mock import MagicMock
from src.llm_selector.selector.auto import AutoSelector
from src.llm_selector.providers.openai import OpenAIProvider
from src.complexity.gemini import GeminiComplexityAnalyzer

@pytest.fixture
def mock_auto_selector():
    """Fixture that initializes AutoSelector with a mocked model registry."""
    performance_config_path = "tests/assests/model_preformance.json"
    analyizer = GeminiComplexityAnalyzer(performance_config_path=performance_config_path)
    selector = AutoSelector(complexity_analyzer=analyizer, performance_config_path=performance_config_path, providers=["openai"])
    selector.select_model = MagicMock(return_value=OpenAIProvider("gpt-3.5-turbo"))
    return selector

def test_select_model(mock_auto_selector):
    """Ensure the selector picks an appropriate model."""
    prompt = "How do black holes form?"
    model = mock_auto_selector.select_model(prompt)
    assert model.model_name == "openai/gpt-3.5-turbo"

def test_generate_response(mock_auto_selector):
    """Ensure the selector generates a response correctly."""
    mock_auto_selector.generate_response = MagicMock(return_value="Black holes form from collapsed stars.")
    response = mock_auto_selector.generate_response("Explain black holes.")
    assert response == "Black holes form from collapsed stars."
