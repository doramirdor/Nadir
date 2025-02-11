from typing import List, Dict, Any
from dataclasses import dataclass, field
from nadir.llm_selector.providers import BaseProvider
from nadir.llm_selector.providers.anthropic import AnthropicProvider
from nadir.llm_selector.providers.openai import OpenAIProvider
from nadir.llm_selector.providers.huggingface import HuggingFaceProvider
from nadir.config.settings import ModelConfig

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
                name="gpt-3o-mini",
                provider="openai",
                complexity_threshold=60.0,
                model_instance=OpenAIProvider("gpt-3o-mini")
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
    
    def get_models_name(self) -> List[str]:
        """
        Get models name
        
        :return: list of models name
        """
        return list(set([model.name for model in self.models]))

    def get_models_provider(self) -> List[str]:
        """
        Get models provider
        
        :return: list of models provider
        """
        return list(set([model.provider for model in self.models]))

    def get_models_full_name(self) -> List[str]:
        """
        Get models provider/name
        
        :return: list of models provider/name
        """
        return list(set([f"{model.provider}/{model.name}" for model in self.models]))


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