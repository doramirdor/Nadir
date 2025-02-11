import logging
from nadir.llm_selector.selector.auto import AutoSelector

class LowLatencySelector(AutoSelector):
    """
    Selects the fastest LLM while dynamically balancing speed and accuracy based on prompt complexity.
    """

    def select_model(self, prompt: str, complexity_details: dict = None):
        """
        Selects the lowest-latency model while considering the complexity of the prompt.
        """
        available_models = self.model_registry.get_sorted_models()
        if complexity_details is None:
            complexity_details = self.complexity_analyzer.get_complexity_details(prompt)

        overall_complexity = complexity_details.get("overall_complexity", 50)  # Default to medium complexity

        # **Dynamic weight adjustment**: Higher complexity → favor accuracy, Lower complexity → favor speed
        SPEED_WEIGHT = max(0.3, 1 - (overall_complexity / 100))  # Lower complexity → Higher weight for speed
        ACCURACY_WEIGHT = min(0.7, overall_complexity / 100)  # Higher complexity → Higher weight for accuracy

        MIN_QUALITY_INDEX = 70  # Minimum acceptable quality for all cases

        # Filter models that meet the minimum quality threshold
        valid_models = [m for m in available_models if float(m.model_instance.metadata.get("Quality Index", 0)) >= MIN_QUALITY_INDEX]

        if not valid_models:
            logging.warning("No models meet the minimum accuracy threshold. Using the fastest available model.")
            valid_models = available_models  # Fallback: Use any model if none meet the threshold

        # Select the best model based on complexity-aware scoring
        best_model = min(
            valid_models,
            key=lambda model: (
                -SPEED_WEIGHT * float(model.model_instance.metadata.get("MedianTokens/s", 1)) +  # Speed priority
                ACCURACY_WEIGHT * float(model.model_instance.metadata.get("Quality Index", 50))  # Accuracy priority
            )
        )

        logging.info(f"Selected low-latency model: {best_model.name} (Complexity: {overall_complexity}, Speed Weight: {SPEED_WEIGHT:.2f}, Accuracy Weight: {ACCURACY_WEIGHT:.2f})")
        return best_model
