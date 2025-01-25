import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("src"), '..')))

from src.complextiy.analyzer import ComplexityAnalyzer

class TestComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = ComplexityAnalyzer()

    def test_simple_prompt_complexity(self):
        simple_prompt = "Hello, how are you?"
        complexity = self.analyzer.calculate_complexity(simple_prompt)
        assert 0 <= complexity <= 100

    def test_complex_prompt_complexity(self):
        complex_prompt = """
        Analyze the multifaceted implications of quantum entanglement 
        on contemporary cryptographic methodologies, exploring the 
        intricate interplay between quantum mechanics and information 
        theory's fundamental principles.
        """
        complexity = self.analyzer.calculate_complexity(complex_prompt)
        assert complexity > 50

    def test_complexity_details(self):
        prompt = "Explain the principles of machine learning algorithms."
        details = self.analyzer.get_complexity_details(prompt)

        assert 'overall_complexity' in details
        assert 'token_count' in details