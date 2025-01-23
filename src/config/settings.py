from typing import Dict, Optional, List
from pydantic import BaseModel, Field, validator
from src.llm_selector.providers import BaseProvider
import os

class ModelConfig(BaseModel):
    """
    Represents configuration for an individual LLM model.
    """
    name: str
    provider: str
    complexity_threshold: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Complexity threshold for selecting the model."
    )
    max_tokens: int = Field(
        default=200000,
        gt=0,
        description="Maximum number of tokens supported by the model."
    )
    cost_per_1k_tokens: float = Field(
        default=0.25,
        gt=0,
        description="Cost per 1000 tokens in USD."
    )
    api_key: Optional[str] = None
    model_instance: BaseProvider = BaseProvider()

    @validator("api_key", always=True)
    def set_api_key_from_env(cls, v, values):
        """
        Dynamically loads the API key for the model based on its provider.
        """
        provider = values.get("provider", "").lower()
        env_var_mapping = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY"
        }
        env_key = env_var_mapping.get(provider)
        return os.getenv(env_key, v) if env_key else v

class DynamicLLMSelectorConfig(BaseModel):
    """
    Manages global configuration for the Dynamic LLM Selector.
    """
    models: Dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "claude-haiku": ModelConfig(
                name="claude-haiku",
                provider="anthropic",
                complexity_threshold=50.0,
                max_tokens=200000,
                cost_per_1k_tokens=0.25
            ),
            "claude-sonnet": ModelConfig(
                name="claude-sonnet",
                provider="anthropic",
                complexity_threshold=75.0,
                max_tokens=200000,
                cost_per_1k_tokens=0.50
            )
        },
        description="Dictionary of registered model configurations."
    )
    complexity_strategy: str = Field(
        default="weighted_average",
        description="Strategy used to calculate model complexity."
    )
    logging_level: str = Field(
        default="INFO",
        description="Logging level for debugging and monitoring."
    )

    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        """
        Retrieves the model configuration by name.

        :param name: The name of the model to retrieve.
        :return: The corresponding ModelConfig or None if not found.
        """
        return self.models.get(name)

    def add_model(self, model_config: ModelConfig):
        """
        Adds a new model to the global configuration.

        :param model_config: The model configuration to register.
        """
        self.models[model_config.name] = model_config

    def validate_configurations(self) -> bool:
        """
        Validates all model configurations in the registry.

        :return: True if all configurations are valid, False otherwise.
        """
        try:
            for model_name, model_config in self.models.items():
                if not model_config.api_key:
                    print(f"Warning: No API key found for model '{model_name}'.")
            return True
        except Exception as e:
            print(f"Error during configuration validation: {e}")
            return False

    def get_models_for_registry(self) -> List[ModelConfig]:
        """
        Returns the list of ModelConfig instances for registration in ModelRegistry.

        :return: List of ModelConfig objects.
        """
        return list(self.models.values())

# Default instance for shared access
DEFAULT_CONFIG = DynamicLLMSelectorConfig()

def load_config_from_env() -> DynamicLLMSelectorConfig:
    """
    Loads the configuration dynamically from environment variables.

    :return: A configured DynamicLLMSelectorConfig instance.
    """
    # Placeholder for more advanced configuration logic
    return DEFAULT_CONFIG
