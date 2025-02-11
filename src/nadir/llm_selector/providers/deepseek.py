from src.nadir.llm_selector.providers import BaseProvider
from typing import Dict, Any

class DeepSeekProvider(BaseProvider):
    def __init__(self, model_name: str = "deepseek-chat"):
        super().__init__(f"deepseek/{model_name}")

    def validate_generation_params(self, params: Dict[str, Any]) -> None:
        """
        Validate generation parameters specific to DeepSeek's API via liteLLM
        """
        supported_params = {
            'model', 'messages', 'temperature', 'max_tokens', 'stream',
            'top_p', 'frequency_penalty', 'presence_penalty', 'stop'
        }
        unsupported = set(params.keys()) - supported_params
        if unsupported:
            raise ValueError(f"Unsupported parameters for DeepSeek: {unsupported}")