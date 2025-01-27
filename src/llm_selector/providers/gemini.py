from src.llm_selector.providers import BaseProvider
from typing import Dict, Any

class GeminiProvider(BaseProvider):
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        super().__init__(f"gemini/{model_name}")


    def validate_generation_params(self, params: Dict[str, Any]) -> None:
        """
        Validate generation parameters specific to Gemini's API via liteLLM
        """
        supported_params = {
            'model', 'messages', 'temperature', 'max_tokens',
            'top_p', 'top_k'
        }
        unsupported = set(params.keys()) - supported_params
        if unsupported:
            raise ValueError(f"Unsupported parameters for Gemini: {unsupported}")