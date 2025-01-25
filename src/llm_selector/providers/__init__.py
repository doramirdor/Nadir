from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for a given prompt
        
        :param prompt: Input prompt
        :param kwargs: Additional generation parameters
        :return: Generated response
        """
        pass

    @abstractmethod
    def generate_with_metadata(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for a given prompt with metadata
        
        :param prompt: Input prompt
        :param kwargs: Additional generation parameters
        :return: Generated response + token sizes
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> int:
        """
        Count tokens in the input text
        
        :param text: Input text
        :return: Token count
        """
        pass

    def validate_generation_params(self, params: Dict[str, Any]) -> None:
        """
        Validate generation parameters
        
        :param params: Generation parameters
        """
        # Default implementation, can be overridden
        pass

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        """
        Provide a minimal schema for the abstract class to avoid recursion.
        """
        # Use a simple schema for the abstract class
        return {"type": "any"}