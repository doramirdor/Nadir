import re
import json
import logging
from typing import Dict, Any, Optional, List
from litellm import completion
from nadir.complexity import BaseComplexityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LLMCodeComplexityAnalyzer(BaseComplexityAnalyzer):
    """
    A complexity analyzer that evaluates code snippets and selects the most suitable model
    based on code structure, execution speed, and cost.

    This class extends `BaseComplexityAnalyzer` and implements its abstract methods.
    """

    def __init__(
        self,
        model_name: str = "gemini/gemini-1.5-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        candidate_names: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        performance_config_path: Optional[str] = None
    ):
        """
        Initialize the complexity analyzer.

        :param model_name: Default LLM model for complexity evaluation.
        :param temperature: Temperature setting for model response.
        :param max_output_tokens: Maximum token limit for output.
        :param candidate_names: Optional list of unique model names to filter candidates.
        :param providers: Optional list of providers (e.g., "OpenAI", "Anthropic").
        :param performance_config_path: Path to the model performance configuration file.
        """
        super().__init__(model_name, candidate_names, providers, performance_config_path)

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        logging.info(f"Initialized LLMCodeComplexityAnalyzer with model: {self.model_name}")

    def calculate_complexity(self, code: str) -> float:
        """
        Compute an overall complexity score (0-100) for the given code snippet.

        :param code: Input code to analyze.
        :return: Complexity score (0-100).
        """
        details = self.get_complexity_details(code)
        complexity_score = details.get("overall_complexity", 0.0)
        logging.info(f"Calculated complexity: {complexity_score}")
        return complexity_score

    def get_complexity_details(self, code: str) -> Dict[str, Any]:
        """
        Analyzes the complexity of a code snippet and selects the best model from available candidates.

        :param code: The input code snippet.
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
            "Your task is to recommend the best model for code complexity analysis by balancing:\n"
            "1️⃣ **Performance:** Evaluate the model's Quality Index, HumanEval, MMLU, and other coding benchmarks.\n"
            "2️⃣ **Speed:** Consider token generation speed, first token response latency, and overall efficiency.\n"
            "3️⃣ **Cost:** Compare the pricing for input/output tokens and blended pricing per million tokens.\n\n"

            "💡 **Optimization Strategy:**\n"
            "- If the code is **long and intricate**, prioritize performance while balancing cost.\n"
            "- If the code is **short and real-time response is key**, prioritize speed over extreme accuracy.\n"
            "- If multiple models have similar accuracy, prefer the best cost-performance-speed balance.\n\n"

            "📌 **Task:** Based on the provided code snippet, determine its complexity and select the best model.\n"
            "Return your answer as a JSON object with:\n"
            "  - `recommended_model`: string (the best model’s unique name)\n"
            "  - `overall_complexity`: number (0-100, representing the complexity of the code)\n"
            "  - `explanation`: string (justification for model selection)\n"
        )

        user_message = f"Code Snippet:\n{code}\n\nWhich model from the above candidates should be used?"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        logging.info(f"Sending code complexity analysis request to {self.model_name}")

        try:
            response = completion(model=self.model_name, messages=messages)
            logging.info("Received model selection response.")
        except Exception as e:
            logging.error(f"Error during model selection: {e}")
            return {"recommended_model": "Unknown", "overall_complexity": -1, "explanation": "API error."}

        raw_text = response.choices[0].message.content
        logging.debug(f"Raw response: {raw_text}")
        return self._parse_json_response(raw_text)

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        """
        Parses the JSON response from the model selection process.

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
