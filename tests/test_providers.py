import pytest
from src.llm_selector.providers.anthropic import AnthropicProvider
from src.llm_selector.providers.openai import OpenAIProvider

class TestProviders:
    def test_anthropic_provider(self):
        provider = AnthropicProvider("claude-3-haiku-20240307")
        response = provider.generate("Hello, world!")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_openai_provider(self):
        provider = OpenAIProvider("gpt-3.5-turbo")
        response = provider.generate("Explain quantum computing")
        assert isinstance(response, str)
        assert len(response) > 0