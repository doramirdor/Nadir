import re
import json
import logging
from typing import Dict, Any, Optional, List
from litellm import completion
from src.complexity import BaseComplexityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GeminiComplexityAnalyzer(BaseComplexityAnalyzer):
    """
    A complexity analyzer that uses Gemini to evaluate the complexity of a prompt.
    It selects the best model based on performance, cost, and speed using data from a configuration file.

    This class extends `BaseComplexityAnalyzer` and implements its abstract methods.
    """

    def __init__(
        self,
        model_name: str = "gemini/gemini-1.5-flash-8b",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        candidate_names: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        performance_config_path: Optional[str] = None
    ):
        """
        Initialize the Gemini complexity analyzer.

        :param model_name: Gemini model to use for evaluation.
        :param temperature: Temperature setting for text generation.
        :param max_output_tokens: Maximum tokens the model should generate.
        :param candidate_names: Optional list of candidate model names to filter.
        :param providers: Optional list of providers (e.g., "OpenAI", "Gemini").
        :param performance_config_path: Path to the model performance JSON file.
        """
        super().__init__(model_name, candidate_names, providers, performance_config_path)

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        logging.info(f"Initialized GeminiComplexityAnalyzer with model: {self.model_name}")

    def calculate_complexity(self, prompt: str) -> float:
        """
        Compute an overall complexity score (0-100) for the given prompt.
        Uses `get_complexity_details` and extracts the score.

        :param prompt: Input text to analyze.
        :return: Complexity score (0-100).
        """
        details = self.get_complexity_details(prompt)
        complexity_score = details.get("overall_complexity", 0.0)
        logging.info(f"Calculated complexity: {complexity_score}")
        return complexity_score

    def get_complexity_details(self, prompt: str) -> Dict[str, Any]:
        """
        Analyzes the complexity of a prompt and selects the best model from available candidates.

        :param prompt: The input text to analyze.
        :return: A dictionary with the recommended model, overall complexity, and explanation.
        """
        if not self.candidate_models:
            logging.warning("No candidate models available; returning default complexity analysis.")
            return {"recommended_model": "Unknown", "overall_complexity": -1, "explanation": "No models available."}

        models_description = self._build_models_description(self.candidate_models)

        system_message = (
            "You are an AI model selection assistant. Below is a list of candidate models with their performance metrics, "
            "pricing information, and speed benchmarks (each identified by its unique name in the format <API Provider>/<Model>):\n\n"
            f"{models_description}\n\n"
            "Your goal is to recommend the best model by balancing three key factors:\n"
            "1️⃣ **Performance:** Evaluate the model's Quality Index, MMLU, HumanEval, and other metrics.\n"
            "2️⃣ **Speed:** Consider token generation speeds and latency.\n"
            "3️⃣ **Cost:** Weigh the blended price per 1M tokens and overall cost efficiency.\n\n"

            "📌 **Task:** Based on the user prompt, analyze the trade-offs between accuracy, cost, and speed, and choose the best model accordingly.\n"
            "Return your response in JSON format with the following keys:\n"
            "  - `recommended_model`: string (the best candidate's unique name)\n"
            "  - `overall_complexity`: number (0-100, reflecting how difficult the prompt is for AI models)\n"
            "  - `explanation`: string (brief reason for selection, highlighting the performance, cost, and speed considerations)\n"
        )

        user_message = f"Prompt:\n{prompt}\n\nWhich model from the above candidates should be used?"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        logging.info(f"Sending model selection prompt to {self.model_name}")

        try:
            response = completion(model=self.model_name, messages=messages)
            logging.info("Received model selection response.")
        except Exception as e:
            logging.error(f"Gemini API error during model selection: {e}")
            return {"recommended_model": "Unknown", "overall_complexity": -1, "explanation": "API error."}

        raw_text = response.choices[0].message.content
        logging.debug(f"Raw model selection response: {raw_text}")
        return self._parse_json_response(raw_text)

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        """
        Cleans and parses a JSON response string.

        :param text: Raw JSON response.
        :return: Parsed dictionary with recommended model, complexity score, and explanation.
        """
        try:
            cleaned_json = re.sub(r'\n', '', re.sub(r'\`', '', text).strip()).replace("json{", "{")
            parsed_response = json.loads(cleaned_json)

            return {
                "recommended_model": parsed_response.get("recommended_model", "Unknown"),
                "overall_complexity": parsed_response.get("overall_complexity", -1),
                "explanation": parsed_response.get("explanation", "No explanation provided.")
            }
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return {"recommended_model": "Unknown", "overall_complexity": -1, "explanation": "Failed to parse response."}

    @staticmethod
    def _build_models_description(candidates: List[Dict[str, Any]]) -> str:
        """
        Constructs a formatted string describing candidate models with their key performance, speed, and cost metrics.

        :param candidates: List of candidate model dictionaries.
        :return: Formatted string describing all models.
        """
        return "\n".join(
            f"{candidate['unique_name']}: "
            f"Quality Index = {candidate.get('Quality Index', 'N/A')}, "
            f"Chatbot Arena = {candidate.get('Chatbot Arena', 'N/A')}, "
            f"MMLU = {candidate.get('MMLU', 'N/A')}, "
            f"GPQA = {candidate.get('GPQA', 'N/A')}, "
            f"MATH-500 = {candidate.get('MATH-500', 'N/A')}, "
            f"HumanEval = {candidate.get('HumanEval', 'N/A')}, "
            f"MedianTokens/s = {candidate.get('MedianTokens/s', 'N/A')}, "
            f"P95Tokens/s = {candidate.get('P95Tokens/s', 'N/A')}, "
            f"Median First Chunk (s) = {candidate.get('Median First Chunk (s)', 'N/A')}, "
            f"Blended Price (USD per 1M tokens) = {candidate.get('Blended Price (USD per 1M tokens)', 'N/A')}, "
            f"Input Price (USD per 1M tokens) = {candidate.get('Input Price (USD per 1M tokens)', 'N/A')}, "
            f"Output Price (USD per 1M tokens) = {candidate.get('Output Price (USD per 1M tokens)', 'N/A')}"
            for candidate in candidates
        )
