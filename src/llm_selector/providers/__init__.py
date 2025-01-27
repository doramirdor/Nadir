from abc import ABC
from typing import Dict, Any
from litellm import completion

class BaseProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for a given prompt using liteLLM
        
        :param prompt: Input prompt
        :param kwargs: Additional generation parameters
        :return: Generated response
        """
        messages = [{"role": "user", "content": prompt}]
        response = completion(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

    def generate_with_metadata(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response for a given prompt with metadata using liteLLM
        
        :param prompt: Input prompt
        :param kwargs: Additional generation parameters
        :return: Dictionary containing generated response and metadata
        """
        messages = [{"role": "user", "content": prompt}]
        response = completion(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return {
            "content": response.choices[0].message.content,
            "usage": response.usage,
            "model": response.model,
            "cost": response.usage.total_cost if hasattr(response.usage, 'total_cost') else None
        }

    def tokenize(self, text: str) -> int:
        """
        Count tokens in the input text using liteLLM's token counter
        
        :param text: Input text
        :return: Token count
        """
        # Note: liteLLM uses tiktoken internally for token counting
        messages = [{"role": "user", "content": text}]
        response = completion(
            model=self.model_name,
            messages=messages,
            max_tokens=0  # Don't generate any tokens, just count
        )
        return response.usage.prompt_tokens

    def validate_generation_params(self, params: Dict[str, Any]) -> None:
        """
        Validate generation parameters for liteLLM
        
        :param params: Generation parameters
        """
        supported_params = {'model', 'messages', 'temperature', 'max_tokens', 'stream'}
        unsupported = set(params.keys()) - supported_params
        if unsupported:
            raise ValueError(f"Unsupported parameters for liteLLM: {unsupported}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        """
        Provide a minimal schema for the abstract class to avoid recursion.
        """
        # Use a simple schema for the abstract class
        return {"type": "any"}