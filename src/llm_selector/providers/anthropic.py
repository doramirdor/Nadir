import anthropic
from src.llm_selector.providers import BaseProvider

class AnthropicProvider(BaseProvider):
    def __init__(self, model_name: str):
        """
        Initialize Anthropic LLM Provider
        
        :param model_name: Anthropic model identifier
        """
        self.client = anthropic.Anthropic()
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Anthropic model
        
        :param prompt: Input prompt
        :param kwargs: Additional generation parameters
        :return: Generated response
        """
        default_params = {
            "model": self.model_name,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        default_params.update(kwargs)
        
        response = self.client.messages.create(
            model=default_params["model"],
            max_tokens=default_params["max_tokens"],
            temperature=default_params["temperature"],
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text

    def tokenize(self, text: str) -> int:
        """
        Count tokens using Anthropic's tokenizer
        
        :param text: Input text
        :return: Token count
        """
        return self.client.count_tokens(text)