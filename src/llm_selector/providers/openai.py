from src.llm_selector.providers import BaseProvider
from typing import Dict, Any

class OpenAIProvider(BaseProvider):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__(f"openai/{model_name}")

    def validate_generation_params(self, params: Dict[str, Any]) -> None:
        """
        Validate generation parameters specific to OpenAI's API via liteLLM
        """
        supported_params = {
            'model', 'messages', 'temperature', 'max_tokens', 'stream',
            'top_p', 'presence_penalty', 'frequency_penalty', 'n', 'stop',
            'logit_bias', 'user'
        }
        unsupported = set(params.keys()) - supported_params
        if unsupported:
            raise ValueError(f"Unsupported parameters for OpenAI: {unsupported}")