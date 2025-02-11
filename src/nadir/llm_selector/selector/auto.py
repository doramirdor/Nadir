
import json
import os
import logging
from typing import Dict, Any, List, Optional
from src.nadir.utils import load_performance_config, safe_float
from src.nadir.complexity import BaseComplexityAnalyzer  
from src.nadir.complexity.llm import LLMComplexityAnalyzer
from src.nadir.llm_selector.model_registry import ModelRegistry
from src.nadir.config.settings import DynamicLLMSelectorConfig, ModelConfig
from src.nadir.llm_selector.providers.openai import OpenAIProvider
from src.nadir.llm_selector.providers.gemini import GeminiProvider
from src.nadir.llm_selector.providers.anthropic import AnthropicProvider
from src.nadir.compression import BaseCompression
from src.nadir.cost.tracker import CostTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# A helper to load model performance configuration from a JSON file.
def build_dynamic_llm_selector_config(json_path: str, providers: List[str] = None) -> DynamicLLMSelectorConfig:
    """
    Builds a DynamicLLMSelectorConfig from a JSON file containing model performance and pricing data.
    """
    candidates = load_performance_config(json_path=json_path)
    models = {}
    for item in candidates:
        # Extract provider and model information
        provider = item.get("api_provider", "")
        model_name = item.get("model", "")
        quality_index = item.get("Quality Index", )
        # Instantiate the appropriate model provider
        if provider == "openai":
            model_instance = OpenAIProvider(model_name)
        elif provider == "gemini":
            model_instance = GeminiProvider(model_name)
        elif provider == "anthropic":
            model_instance = AnthropicProvider(model_name)
        else:
            model_instance = None
            continue # we currently not supporitng other models for auto selection

        if providers and provider not in providers:
            continue
        # Create and store the ModelConfig
        models[model_name] = ModelConfig(
            name=model_name,
            provider=provider,
            complexity_threshold=quality_index,
            model_instance=model_instance
        )

    return DynamicLLMSelectorConfig(models=models)


class AutoSelector:
    """
    AutoSelector dynamically selects an LLM model for a given prompt based on its complexity,
    optimizing for performance, speed, and cost.

    Features:
    - Model selection based on prompt complexity
    - Supports multiple providers (OpenAI, Gemini, Anthropic)
    - Integrated cost tracking for monitoring expenses
    - Response compression for efficient token usage
    """

    def __init__(
        self,
        complexity_analyzer: Optional[BaseComplexityAnalyzer] = None,
        model_registry: Optional[ModelRegistry] = None,
        compression: Optional[BaseCompression] = None,
        logger: Optional[logging.Logger] = None,
        dynamic_config: Optional[DynamicLLMSelectorConfig] = None,
        performance_config_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ):
        """
        Initializes the AutoSelector with model selection capabilities.

        :param complexity_analyzer: The complexity analyzer for evaluating prompt complexity.
        :param model_registry: The registry managing available models.
        :param compression: Optional compression module for reducing token usage.
        :param logger: Logger instance for logging.
        :param dynamic_config: Preloaded configuration for models.
        :param performance_config_path: Path to the JSON file with model performance details.
        :param providers: List of providers to filter available models.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.compression = compression or BaseCompression()
        self.cost_tracker = CostTracker()

        # Load model performance configuration if not provided
        if dynamic_config is None:
            dynamic_config = self._load_performance_config(performance_config_path, providers)

        # Initialize the model registry
        self.model_registry = model_registry or ModelRegistry()
        if model_registry is None:
            self.model_registry.register_models(dynamic_config.get_models_for_registry())

        # Get list of models for complexity analysis
        models_api_name = self.model_registry.get_models_full_name()
        self.complexity_analyzer = complexity_analyzer or LLMComplexityAnalyzer(candidate_names=models_api_name)

        self.logger.info(f"AutoSelector initialized with {len(self.model_registry.get_sorted_models())} models.")

    def _load_performance_config(
        self, performance_config_path: Optional[str], providers: Optional[List[str]]
    ) -> DynamicLLMSelectorConfig:
        """Loads model performance configuration from a JSON file or falls back to defaults."""
        if performance_config_path is None:
            performance_config_path = os.getenv(
                "MODEL_PERFORMANCE_CONFIG",
                os.path.join(os.path.dirname(__file__), "../../config/model_performance.json")
            )

        try:
            dynamic_config = build_dynamic_llm_selector_config(performance_config_path, providers)
            self.logger.info(f"Loaded model performance config from {performance_config_path}")
            return dynamic_config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}. Falling back to default models.")
            return self._default_model_config()

    def _default_model_config(self) -> DynamicLLMSelectorConfig:
        """Provides a default model configuration in case of failure."""
        return DynamicLLMSelectorConfig(models={
            "gpt-3.5-turbo": ModelConfig(
                name="gpt-3.5-turbo",
                provider="openai",
                complexity_threshold=50.0,
                model_instance=OpenAIProvider("gpt-3.5-turbo")
            ),
            "gpt-4o-mini": ModelConfig(
                name="gpt-4o-mini",
                provider="openai",
                complexity_threshold=75.0,
                model_instance=OpenAIProvider("gpt-4o-mini")
            ),
            "gpt-4o": ModelConfig(
                name="gpt-4o",
                provider="openai",
                complexity_threshold=100.0,
                model_instance=OpenAIProvider("gpt-4o")
            )
        })

    def select_model(self, prompt: str, complexity_details: Optional[Dict[str, Any]] = None) -> ModelConfig:
        """
        Selects the best model based on prompt complexity.

        :param prompt: Input text for evaluation.
        :param complexity_details: Optional precomputed complexity details.
        :return: The most suitable ModelConfig object.
        """
        try:
            if complexity_details is None:
                complexity_details = self.complexity_analyzer.get_complexity_details(prompt)

            available_models = self.model_registry.get_sorted_models()

            for model in available_models:
                if complexity_details["recommended_model"] == f"{model.provider}/{model.name}":
                    self.logger.info(f"Selected model: {model.name}")
                    return model

            self.logger.warning(f"No exact match found. Falling back to {available_models[0].name}.")
            return available_models[0]

        except Exception as e:
            self.logger.error(f"Model selection error: {e}. Falling back to default.")
            return self._default_model_config().models["gpt-3.5-turbo"]

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generates a response using the best-selected model.

        :param prompt: Input prompt.
        :return: Model-generated response.
        """
        try:
            max_tokens = kwargs.pop("max_tokens_for_compression", -1)
            compressed_prompt = self.compression.compress(prompt, max_tokens) if max_tokens != -1 else self.compression.compress(prompt)

            selected_model = self.select_model(compressed_prompt)
            if not selected_model.model_instance:
                raise ValueError(f"No model instance found for {selected_model.name}")

            response = selected_model.model_instance.generate_with_metadata(compressed_prompt, **kwargs)
            self.logger.info(f"Response generated using {selected_model.name}")

            usage = response["usage"]
            self.cost_tracker.add_cost(
                selected_model.model_instance.model_name,
                usage["prompt_tokens"],
                usage["completion_tokens"]
            )

            return response["content"]

        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            raise

    def get_complexity_details(self, prompt: str) -> Dict[str, Any]:
        """
        Computes complexity details for a prompt and selects the most appropriate model.

        :param prompt: Input text.
        :return: A dictionary with complexity score, recommended model, and details.
        """
        try:
            complexity_details = self.complexity_analyzer.get_complexity_details(prompt)
            selected_model = self.select_model(prompt, complexity_details)

            return {
                "prompt": prompt,
                "complexity_score": complexity_details['overall_complexity'],
                "selected_model": selected_model.name,
                "details": complexity_details
            }

        except Exception as e:
            self.logger.error(f"Complexity analysis error: {e}")
            raise

    def list_available_models(self) -> List[str]:
        """
        Lists available models sorted by complexity threshold.

        :return: List of model names.
        """
        return [model.name for model in self.model_registry.get_sorted_models()]
