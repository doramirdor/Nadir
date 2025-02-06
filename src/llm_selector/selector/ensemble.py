import logging
from collections import Counter
from src.llm_selector.selector.auto import AutoSelector

class EnsembleSelector(AutoSelector):
    """
    Uses multiple models to generate responses and selects the most commonly recommended one.
    It ensures the selected models match the prompt's complexity.
    """

    def select_model(self, prompt: str, complexity_details: dict = None):
        """
        Runs exactly 3 models that match the complexity level and selects the most commonly recommended one.
        """
        available_models = self.model_registry.get_sorted_models()

        if complexity_details is None:
            complexity_details = self.complexity_analyzer.get_complexity_details(prompt)

        overall_complexity = complexity_details.get("overall_complexity", 50)  # Default to medium complexity

        # **Step 1: Filter models that match the complexity level**
        complexity_matched_models = [
            model for model in available_models
            if abs(overall_complexity - float(model.model_instance.metadata.get("Quality Index", 50))) <= 15
        ]

        if len(complexity_matched_models) < 3:
            logging.warning(f"Not enough models match complexity {overall_complexity}. Using closest available models.")
            complexity_matched_models = available_models[:3]  # Fallback: Take the top 3 models

        # **Step 2: Select exactly 3 models for ensemble**
        selected_models = complexity_matched_models[:3]
        logging.info(f"Running ensemble with models: {[m.name for m in selected_models]} (Complexity: {overall_complexity})")

        # **Step 3: Get responses from the selected models**
        model_votes = []
        for model in selected_models:
            response = model.model_instance.generate_with_metadata(prompt)
            model_votes.append(response.get("model"))

        # **Step 4: Perform majority voting**
        most_voted_model = Counter(model_votes).most_common(1)[0][0]

        logging.info(f"Ensemble selected model: {most_voted_model} (Complexity: {overall_complexity})")
        return self.model_registry.get_model_by_name(most_voted_model)
