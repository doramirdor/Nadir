from typing import List, Dict, Any
from dataclasses import dataclass, field
from src.llm_selector.providers import BaseProvider
from src.llm_selector.providers.anthropic import AnthropicProvider
from src.llm_selector.providers.openai import OpenAIProvider
from src.llm_selector.providers.huggingface import HuggingFaceProvider

@dataclass
class ModelConfig:
    name: str
    provider: str
    complexity_threshold: float = 50.0
    max_tokens: int = 200000
    cost_per_1k_tokens: float = 0.25
    model_instance: Any = None

class ModelRegistry:
    def __init__(self):
        self.models: List[ModelConfig] = []
        self._default_registration()

    def _default_registration(self):
        """
        Register default models from different providers
        """
        default_models = [
            ModelConfig(
                name="gpt-3.5-turbo",
                provider="openai",
                complexity_threshold=40.0,
                model_instance=OpenAIProvider("gpt-3.5-turbo")
            ),
            ModelConfig(
                name="gpt-4",
                provider="openai",
                complexity_threshold=60.0,
                model_instance=OpenAIProvider("gpt-4")
            )
        ]
        
        self.models.extend(default_models)

    def register_model(self, model_config: ModelConfig):
        """
        Register a new model in the registry
        
        :param model_config: Model configuration
        """
        self.models.append(model_config)

    def register_models(self, model_configs: List[ModelConfig]):
        """
        Register multiple models in the registry
        
        :param model_configs: List of ModelConfig objects to register
        """
        self.models = []
        for model_config in model_configs:
            self.register_model(model_config)

    def get_sorted_models(self) -> List[ModelConfig]:
        """
        Get models sorted by complexity threshold
        
        :return: Sorted list of models
        """
        return sorted(
            self.models, 
            key=lambda x: x.complexity_threshold
        )

    def get_model_by_name(self, name: str) -> ModelConfig:
        """
        Get a specific model by name
        
        :param name: Model name
        :return: Model configuration
        """
        for model in self.models:
            if model.name == name:
                return model
        raise ValueError(f"Model {name} not found")