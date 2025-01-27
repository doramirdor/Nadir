from src.llm_selector.providers import BaseProvider
from typing import Dict, Any


class AnthropicProvider(BaseProvider):
    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        super().__init__(model_name)

    def validate_generation_params(self, params: Dict[str, Any]) -> None:
        """
        Validate generation parameters specific to Anthropic's API via liteLLM
        """
        supported_params = {
            'model', 'messages', 'temperature', 'max_tokens', 'stream',
            'top_p', 'top_k', 'stop_sequences', 'metadata'
        }
        unsupported = set(params.keys()) - supported_params
        if unsupported:
            raise ValueError(f"Unsupported parameters for Anthropic: {unsupported}")