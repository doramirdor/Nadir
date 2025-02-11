import logging
from src.nadir.llm_selector.selector.auto import AutoSelector

class BalancedSelector(AutoSelector):
    """
    Finds a model that provides a good balance between accuracy, speed, and cost,
    while ensuring it aligns with the prompt's complexity level.
    """

    def select_model(self, prompt: str, complexity_details: dict = None):
        """Uses a weighted scoring system to find the best overall model."""
        available_models = self.model_registry.get_sorted_models()

        if complexity_details is None:
            complexity_details = self.complexity_analyzer.get_complexity_details(prompt)

        overall_complexity = complexity_details.get("overall_complexity", 50)  # Default to medium complexity

        # **Step 1: Filter models that match the complexity level**
        complexity_matched_models = [
            model for model in available_models
            if abs(overall_complexity - float(model.model_instance.metadata.get("Quality Index", 50))) <= 15
        ]

        if not complexity_matched_models:
            logging.warning(f"No models match complexity {overall_complexity}. Using closest available models.")
            complexity_matched_models = available_models  # Fallback: Use all models

        # **Step 2: Define the scoring function**
        def model_score(model):
            quality = float(model.model_instance.metadata.get("Quality Index", 50))
            speed = float(model.model_instance.metadata.get("MedianTokens/s", 1))
            cost = float(model.model_instance.metadata.get("Blended Price (USD per 1M tokens)", 100))

            # Adjust weights dynamically based on complexity
            quality_weight = 0.5 if overall_complexity > 60 else 0.3
            speed_weight = 0.3 if overall_complexity < 40 else 0.2
            cost_weight = 0.2 if overall_complexity > 80 else 0.5

            return (quality_weight * quality) + (speed_weight * speed) - (cost_weight * cost)

        # **Step 3: Select the best model based on scoring**
        best_model = max(complexity_matched_models, key=model_score)

        logging.info(f"Selected balanced model: {best_model.name} (Complexity: {overall_complexity})")
        return best_model
