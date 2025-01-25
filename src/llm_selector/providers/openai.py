import os
import openai
from src.llm_selector.providers import BaseProvider

class OpenAIProvider(BaseProvider):
    def __init__(self, model_name: str):
        """
        Initialize OpenAI LLM Provider
        
        :param model_name: OpenAI model identifier (e.g., "gpt-3.5-turbo", "gpt-4", etc.)
        """
        self.model_name = model_name
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # If you use Azure OpenAI, you might need something like:
        # openai.api_base = os.getenv("OPENAI_API_BASE", "<YOUR_AZURE_ENDPOINT>")
        # openai.api_type = "azure"
        # openai.api_version = "2023-07-01"  # or the version you require

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using the OpenAI ChatCompletion endpoint.
        
        :param prompt: Input prompt or text
        :param kwargs: Additional generation parameters
        :return: Generated response as a string
        """
        default_params = {
            "model": self.model_name,
            "max_tokens": 1000,
            "temperature": 0.7,
        }
        default_params.update(kwargs)
        
        response = openai.chat.completions.create(
            model=default_params["model"],
            messages=[{
                "role": "user",
                "content": prompt
                }],
            max_tokens=default_params["max_tokens"],
            temperature=default_params["temperature"]
            )
        return response.choices[0].message.content

    def tokenize(self, text: str) -> int:
        """
        Count tokens using tiktoken. If the model is not recognized,
        fall back to a default encoding (cl100k_base).
        
        :param text: Input text
        :return: Number of tokens
        """
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback if model is unknown to tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    
    def generate_with_metadata(self, prompt: str, **kwargs) -> str:
        """
        Generate a response and calculate the cost based on token usage.
        
        :param prompt: Input prompt or text
        :param kwargs: Additional generation parameters
        :return: A dictionary with the response and usage details
        """
        default_params = {
            "model": self.model_name,
            "max_tokens": 1000,
            "temperature": 0.7,
        }
        default_params.update(kwargs)
        
        response = openai.chat.completions.create(
            model=default_params["model"],
            messages=[{
                "role": "user",
                "content": prompt
                }],
            max_tokens=default_params["max_tokens"],
            temperature=default_params["temperature"]
            )
        
        # Extract token usage
        usage = response.usage  # Access the 'usage' attribute directly
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        return {
                "response": response.choices[0].message.content,
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }


    def generate_response_with_cost(self, prompt: str, **kwargs) -> dict:
        """
        Generate a response and calculate the cost based on token usage and pricing.
        
        :param prompt: Input prompt or text
        :param kwargs: Additional generation parameters
        :return: A dictionary with the response and cost details
        """
        # Pricing details per 1M tokens (update if OpenAI changes prices)
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.006},
            "gpt-4": {"input": 0.01, "output": 0.03},
            "gpt-4-32k": {"input": 0.03, "output": 0.06},
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "o1": {"input": 0.015, "output": 0.06},
            "o1-mini": {"input": 0.003, "output": 0.012},
            "gpt-4o-realtime-preview": {"input": 0.005, "output": 0.02},
            "gpt-4o-mini-realtime-preview": {"input": 0.0006, "output": 0.0024},
        }

        default_params = {
            "model": self.model_name,
            "max_tokens": 1000,
            "temperature": 0.7,
        }
        default_params.update(kwargs)

        try:
            response = openai.chat.completions.create(
                model=default_params["model"],
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=default_params["max_tokens"],
                temperature=default_params["temperature"]
            )

            # Extract token usage
            usage = response["usage"]
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            total_tokens = usage["total_tokens"]

            # Get pricing for the model
            model_pricing = pricing.get(self.model_name)
            if not model_pricing:
                raise ValueError(f"Pricing information not available for model: {self.model_name}")

            # Calculate costs
            input_cost = (prompt_tokens / 1000000) * model_pricing["input"]
            output_cost = (completion_tokens / 1000000) * model_pricing["output"]
            total_cost = input_cost + output_cost

            return {
                "response": response.choices[0].message.content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost
            }
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")
