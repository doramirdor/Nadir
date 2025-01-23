import os
import google.generativeai as genai
from typing import Optional, Dict, Any

from src.llm_selector.providers import BaseProvider

class GeminiProvider(BaseProvider):
    def __init__(
        self, 
        model_name: str = "gemini-pro", 
        api_key: Optional[str] = None
    ):
        """
        Initialize Gemini LLM Provider
        
        :param model_name: Gemini model identifier
        :param api_key: Google API key
        """
        # Use provided API key or load from environment
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            raise ValueError("Google API key is required")
        
        # Configure Google AI library
        genai.configure(api_key=api_key)
        
        # Select model
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Gemini model
        
        :param prompt: Input prompt
        :param kwargs: Additional generation parameters
        :return: Generated response
        """
        # Default generation parameters
        generation_config = {
            'temperature': 0.7,
            'max_output_tokens': 1024,
            **kwargs
        }
        
        try:
            response = self.model.generate_content(
                prompt, 
                generation_config=generation_config
            )
            
            return response.text
        
        except Exception as e:
            raise RuntimeError(f"Gemini generation error: {e}")

    def tokenize(self, text: str) -> int:
        """
        Estimate token count for Gemini
        
        :param text: Input text
        :return: Estimated token count
        """
        # Gemini uses a different tokenization method
        # This is a rough approximation
        return len(text.split())