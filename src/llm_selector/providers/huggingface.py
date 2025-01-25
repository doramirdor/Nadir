import os
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.llm_selector.providers import BaseProvider

class HuggingFaceProvider(BaseProvider):
    def __init__(
        self, 
        model_name: str, 
        max_length: int = 512,
        temperature: float = 0.5
    ):
        """
        Initialize Hugging Face LLM Provider
        
        :param model_name: Hugging Face model identifier
        :param max_length: Maximum token length for generation
        :param temperature: Sampling temperature for generation
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set the pad token to EOS if necessary (GPT-2 doesn't have a pad token by default).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Hugging Face model
        
        :param prompt: Input prompt
        :param kwargs: Additional generation parameters
        :return: Generated response
        """
        # Merge default and provided parameters
        generation_params = {
            'max_length': self.max_length,
            'temperature': self.temperature,
            'do_sample': True,                # Enable sampling so temperature matters
            'top_p': 0.95,                    # Nucleus sampling; adjust as needed
            'repetition_penalty': 1.2,        # Helps reduce repeated lines
            'pad_token_id': self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Tokenize input (include attention mask)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            return_attention_mask=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask,
                **generation_params
            )
        
        # Decode and return response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Optionally remove the prompt if it's repeated at the start of the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    

    def generate_with_metadata(self, prompt: str, **kwargs) -> dict:
        """
        Generate response with metadata using Hugging Face model.
        
        :param prompt: Input prompt
        :param kwargs: Additional generation parameters
        :return: Dictionary containing response and token usage
        """
        # Merge default and provided parameters
        generation_params = {
            'max_length': self.max_length,
            'temperature': self.temperature,
            'do_sample': True,
            'top_p': 0.95,
            'repetition_penalty': 1.2,
            'pad_token_id': self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Tokenize input (include attention mask)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            return_attention_mask=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask,
                **generation_params
            )
        
        # Decode response
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the response if repeated
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].strip()

        # Token usage
        prompt_tokens = len(inputs.input_ids[0])  # Token count for the prompt
        completion_tokens = len(self.tokenizer.encode(response_text))  # Token count for the generated response
        total_tokens = prompt_tokens + completion_tokens

        return {
            "response": response_text,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

    def tokenize(self, text: str) -> int:
        """
        Count tokens using Hugging Face tokenizer
        
        :param text: Input text
        :return: Token count
        """
        return len(self.tokenizer.encode(text))
