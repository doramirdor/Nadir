from typing import Dict, Any, List, Optional
from src.complextiy import BaseComplexityAnalyzer
from src.complextiy.analyzer import ComplexityAnalyzer
from src.llm_selector.model_registry import ModelRegistry
from src.config.settings import ModelConfig
from src.llm_selector.providers import BaseProvider
from src.compression import BaseCompression
from src.cost.tracker import CostTracker
from src import logging

class LLMSelector:
    def __init__(
        self, 
        complexity_analyzer: Optional[BaseComplexityAnalyzer] = None,
        model_registry: Optional[ModelRegistry] = None,
        compression: Optional[BaseCompression] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize LLM Selector with complexity analysis, model registry, and optional compression.
        
        :param complexity_analyzer: Custom complexity analyzer
        :param model_registry: Custom model registry
        :param compression: An optional compression strategy (defaults to BaseCompression, which does nothing).
        :param logger: Custom logger
        """
        self.complexity_analyzer = complexity_analyzer or ComplexityAnalyzer()
        self.model_registry = model_registry or ModelRegistry()
        self.compression = compression or BaseCompression()
        self.logger = logger or logging.getLogger(__name__)
        self.cost_tracker = CostTracker()  # Add CostTracker

    def select_model(self, prompt: str) -> ModelConfig:
        """
        Dynamically select the most appropriate model for a given prompt based on complexity.
        
        :param prompt: Input prompt
        :return: Selected model configuration
        """
        try:
            complexity = self.complexity_analyzer.calculate_complexity(prompt)
            self.logger.info(f"Prompt complexity: {complexity}")
            
            # Get models sorted by complexity threshold
            available_models = self.model_registry.get_sorted_models()
            
            # Find the first model that can handle this complexity
            for model in available_models:
                if complexity <= model.complexity_threshold:
                    self.logger.info(f"Selected model: {model.name}")
                    return model
            
            # Default to the most capable model if none matched
            default_model = available_models[-1]
            self.logger.warning(f"No model found for complexity {complexity}. "
                                f"Defaulting to {default_model.name}.")
            return default_model
        
        except Exception as e:
            self.logger.error(f"Error in model selection: {e}")
            raise

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using a dynamically selected model.
        If compression is configured, it will be applied first.
        
        :param prompt: Input prompt
        :param kwargs: Additional generation parameters
        :return: Generated response
        """
        try:
            # Compress the prompt if needed
            max_tokens_for_compression = kwargs.pop("max_tokens_for_compression", -1)
            if max_tokens_for_compression == -1:
                # No token limit specified, call compress without second argument
                compressed_prompt = self.compression.compress(prompt)
            else:
                compressed_prompt = self.compression.compress(prompt, max_tokens_for_compression)
                    
            selected_model = self.select_model(compressed_prompt)
            if not selected_model.model_instance:
                raise ValueError(f"No model instance for {selected_model.name}")
            
            response = selected_model.model_instance.generate_with_metadata(compressed_prompt, **kwargs)
            self.logger.info(f"Response generated using {selected_model.name}, response: {response}")

            usage = response["usage"]
            self.cost_tracker.add_cost(selected_model.model_instance.model_name, usage["prompt_tokens"], usage["completion_tokens"])

            return response["content"]
        
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            raise

    def get_complexity_details(self, prompt: str) -> Dict[str, Any]:
        """
        Get a detailed complexity analysis for a prompt and show which model it would select.
        
        :param prompt: Input text
        :return: A dictionary with complexity details and chosen model info
        """
        try:
            complexity_details = self.complexity_analyzer.get_complexity_details(prompt)
            selected_model = self.select_model(prompt)
            
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
        Return a list of model names, sorted by ascending complexity threshold.
        
        :return: List of model names
        """
        return [model.name for model in self.model_registry.get_sorted_models()]