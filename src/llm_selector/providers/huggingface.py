from src.llm_selector.providers import BaseProvider
from typing import Dict, Any


class HuggingFaceProvider(BaseProvider):
    def __init__(self, model_name: str = "mistral-7b-instruct-v0.2"):
        super().__init__(f"huggingface/{model_name}")

    def validate_generation_params(self, params: Dict[str, Any]) -> None:
        """
        Validate generation parameters specific to HuggingFace models via liteLLM
        """
        supported_params = {
            'model', 'messages', 'temperature', 'max_tokens', 'stream',
            'top_p', 'top_k', 'repetition_penalty', 'stop'
        }
        unsupported = set(params.keys()) - supported_params
        if unsupported:
            raise ValueError(f"Unsupported parameters for HuggingFace: {unsupported}")