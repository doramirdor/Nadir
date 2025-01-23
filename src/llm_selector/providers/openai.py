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
            # You can add more params here if desired: top_p, presence_penalty, etc.
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
