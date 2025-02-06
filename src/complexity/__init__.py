import os
import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from src.utils import load_performance_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BaseComplexityAnalyzer(ABC):
    """
    Abstract base class for complexity analyzers.

    This class defines the structure for complexity analyzers that evaluate the complexity
    of input prompts or code snippets and select the best model based on performance, cost, and speed.

    Implementing classes must define `calculate_complexity`, `get_complexity_details`, 
    and `_parse_json_response` methods.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        candidate_names: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        performance_config_path: Optional[str] = None
    ):
        """
        Initialize the complexity analyzer with candidate models.

        :param model_name: Optional name of the default model.
        :param candidate_names: Optional list of specific candidate models.
        :param providers: Optional list of provider names (e.g., "OpenAI", "Gemini").
        :param performance_config_path: Path to the model performance JSON file.
        """
        self.model_name = model_name

        # Load performance configuration file
        if performance_config_path is None:
            performance_config_path = os.getenv(
                "MODEL_PERFORMANCE_CONFIG",
                os.path.join(os.path.dirname(__file__), "../config/model_performance.json")
            )

        try:
            full_candidates = load_performance_config(performance_config_path)
            logging.info(f"Loaded {len(full_candidates)} candidates from {performance_config_path}")
        except Exception as e:
            logging.error(f"Error loading performance config from {performance_config_path}: {e}")
            full_candidates = []

        self.candidate_models = self._filter_candidates(full_candidates, candidate_names, providers)

        logging.info(f"Initialized {self.__class__.__name__} with {len(self.candidate_models)} candidate(s).")

    def _filter_candidates(
        self, full_candidates: List[Dict[str, Any]], candidate_names: Optional[List[str]], providers: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Filters candidates based on model names and providers.

        :param full_candidates: List of all available candidate models.
        :param candidate_names: Specific models to include.
        :param providers: Specific providers to include.
        :return: Filtered list of candidates.
        """
        if providers:
            return [c for c in full_candidates if c["api_provider"] in providers]
        elif candidate_names:
            return [c for c in full_candidates if c["unique_name"] in candidate_names]
        return full_candidates

    @abstractmethod
    def calculate_complexity(self, input_text: str) -> float:
        """
        Compute an overall complexity score (0-100) based on multiple factors.
        Must be implemented in subclasses.

        :param input_text: The input (prompt or code snippet) to analyze.
        :return: A numeric complexity score between 0 and 100.
        """
        pass

    @abstractmethod
    def get_complexity_details(self, input_text: str) -> Dict[str, Any]:
        """
        Provide a detailed breakdown of the complexity metrics.
        Must be implemented in subclasses.

        :param input_text: The input (prompt or code snippet) to analyze.
        :return: A dictionary containing various complexity metrics.
        """
        pass

    @staticmethod
    @abstractmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        """
        Parses and cleans a JSON response string.
        Must be implemented in subclasses.

        :param text: Raw JSON response as a string.
        :return: Parsed dictionary.
        """
        pass
