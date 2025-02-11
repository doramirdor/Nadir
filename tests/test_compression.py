import pytest
from unittest.mock import MagicMock
from src.nadir.llm_selector.providers.openai import OpenAIProvider

@pytest.fixture
def mock_openai_provider():
    provider = OpenAIProvider("gpt-3.5-turbo")
    provider.generate = MagicMock(return_value="Hello, world!")
    return provider

def test_openai_provider_generate(mock_openai_provider):
    """Test if OpenAIProvider generates a valid response."""
    response = mock_openai_provider.generate("Say hello")
    assert response == "Hello, world!"
