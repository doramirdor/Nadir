import re
import json
import os
from typing import Dict, Any, Optional

import google.generativeai as genai

from src.complextiy import BaseComplexityAnalyzer

class GeminiComplexityAnalyzer(BaseComplexityAnalyzer):
    """
    A complexity analyzer that uses Google's Gemini 1.5 Flash model
    to analyze and score the complexity of a given prompt.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 1024
    ):
        """
        :param api_key: Your Google Generative AI API key (fallback to env var if None).
        :param model_name: The Gemini model (default: gemini-1.5-flash).
        :param temperature: Controls randomness of generation.
        :param max_output_tokens: Maximum tokens in the response.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("A Google Generative AI API key is required.")

        # Configure the generative AI library
        genai.configure(api_key=self.api_key)

        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def calculate_complexity(self, prompt: str) -> float:
        """
        Returns a single numerical score (0-100) representing the LLM's assessment.
        """
        response_data = self.get_complexity_details(prompt)
        return response_data.get("overall_complexity", 0.0)

    def get_complexity_details(self, prompt: str) -> Dict[str, Any]:
        """
        Sends the prompt to Gemini with instructions to produce:
          - overall_complexity (0 to 100)
          - an explanation
        Returns a dict: { overall_complexity: float, explanation: str, ... }
        """
        system_message = (
            "You are a complexity assessor. When given some text, you will:\n"
            "1. Provide an overall complexity score from 0 to 100 (0=trivial, 100=extremely complex).\n"
            "2. Briefly explain how you arrived at that score.\n\n"
            "Return your answer as valid JSON with keys:\n"
            "- overall_complexity: number\n"
            "- explanation: string\n"
        )

        user_message = f"TEXT:\n{prompt}\n\nWhat is the overall complexity score from 0-100? Provide a short explanation."

        full_prompt = f"{system_message}\n\n{user_message}"

        try:
            # Generate with Gemini
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens
                }
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

        # The response is typically in response.text
        raw_text = response.text.strip()

        return self._parse_json_response(raw_text)


    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Attempt to parse LLM output as JSON. 
        1. First, search for a code block enclosed by ```json ... ``` 
        2. If found, parse only that block. Otherwise parse the entire text.
        3. If invalid JSON, fallback with overall_complexity=-1 and explanation as raw text.
        """
        # Regex to find ```json ... ```
        pattern = r'```json\s*(.*?)\s*```'
        match = re.search(pattern, text, flags=re.DOTALL)

        # If a code block is found, parse it; otherwise, parse the entire text.
        json_str = match.group(1) if match else text

        try:
            parsed = json.loads(json_str)

            # Ensure required keys exist
            if "overall_complexity" not in parsed:
                parsed["overall_complexity"] = -1
            if "explanation" not in parsed:
                parsed["explanation"] = ""
            return parsed
        except json.JSONDecodeError:
            # Fallback if the content is not valid JSON
            return {
                "overall_complexity": -1,
                "explanation": text
            }

