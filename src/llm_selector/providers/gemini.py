import os
import yaml
import google.generativeai as genai
from typing import Optional, Dict, Any

from src.llm_selector.providers import BaseProvider

class GeminiProvider(BaseProvider):
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Gemini LLM Provider
        
        :param model_name: Gemini model identifier
        :param api_key: Google API key
        :param pricing_file: Path to YAML file with pricing details
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
        """Estimate token count for Gemini."""
        return len(text.split())
    
    def generate_with_metadata(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Gemini model with metadata
        
        :param prompt: Input prompt
        :param kwargs: Additional generation parameters
        :return: Generated response
        """
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
            return   {
                "response": response.text,
                "usage": {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            }
        
        
        except Exception as e:
            raise RuntimeError(f"Gemini generation error: {e}")

    def generate_response_with_metadata(self, prompt: str, **kwargs) -> dict:
        """
        Generate a response and calculate the cost.
        
        :param prompt: Input prompt
        :param kwargs: Additional generation parameters
        :return: Dictionary with response and cost
        """
        generation_config = {
            'temperature': 0.7,
            'max_output_tokens': 1024,
            **kwargs
        }
        
        try:
            # Generate content
            response = self.model.generate_content(
                prompt, 
                generation_config=generation_config
            )
            
            # Extract token counts
            prompt_tokens = response.result.usage_metadata["prompt_token_count"]
            candidate_tokens = response.result.usage_metadata["candidates_token_count"]
            total_tokens = response.result.usage_metadata["total_token_count"]
            
            # Determine pricing model
            model_pricing = self.pricing.get(self.model_name.lower().replace("-", "_"))
            
            if not model_pricing:
                raise ValueError(f"Pricing details not found for model: {self.model_name}")
            
            # Calculate input and output costs
            input_pricing = model_pricing["input_pricing"]["below_128k"]
            output_pricing = model_pricing["output_pricing"]["below_128k"]
            input_cost = (prompt_tokens / 1_000_000) * input_pricing
            output_cost = (candidate_tokens / 1_000_000) * output_pricing
            
            # Total cost
            total_cost = input_cost + output_cost
            
            return {
                "response": response.result["candidates"][0]["content"]["parts"][0]["text"],
                "cost": {
                    "input_cost": round(input_cost, 4),
                    "output_cost": round(output_cost, 4),
                    "total_cost": round(total_cost, 4),
                }
            }
        
        except Exception as e:
            raise RuntimeError(f"Error in generate_response_with_cost: {e}")
