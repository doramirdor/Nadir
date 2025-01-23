import pytest
from src.llm_selector.core import LLMSelector
from src.llm_selector.model_registry import ModelRegistry

class TestModelSelection:
    def setup_method(self):
        self.selector = LLMSelector()

    def test_model_selection(self):
        simple_prompt = "Write a short greeting."
        selected_model = self.selector.select_model(simple_prompt)
        assert selected_model.name in self.selector.list_available_models()

    def test_complex_prompt_model_selection(self):
        complex_prompt = """
        Provide an in-depth analysis of quantum computing's 
        potential revolutionary impact on cryptographic systems, 
        discussing both theoretical foundations and practical implications.
        """
        selected_model = self.selector.select_model(complex_prompt)
        assert selected_model.complexity_threshold >= 75