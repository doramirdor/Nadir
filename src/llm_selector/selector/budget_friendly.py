import logging
from src.llm_selector.selector.auto import AutoSelector

class BudgetFriendlySelector(AutoSelector):
    """
    Selects the most cost-effective model while ensuring it meets the complexity level.
    Useful for large-scale applications where cost reduction is crucial.
    """

    def select_model(self, prompt: str, complexity_details: dict = None):
        """
        Selects the lowest-cost model while still meeting complexity requirements.
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

        if not complexity_matched_models:
            logging.warning(f"No models match complexity {overall_complexity}. Using closest available models.")
            complexity_matched_models = available_models  # Fallback: Use all models

        # **Step 2: Select the most cost-effective model among the filtered ones**
        cheapest_model = min(
            complexity_matched_models,
            key=lambda model: float(model.model_instance.metadata.get("Blended Price (USD per 1M tokens)", 100))
        )

        logging.info(f"Selected budget-friendly model: {cheapest_model.name} (Complexity: {overall_complexity})")
        return cheapest_model
