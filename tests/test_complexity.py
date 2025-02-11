import pytest
from nadir.complexity.analyzer import ComplexityAnalyzer

@pytest.fixture
def complexity_analyzer():
    """Fixture to provide a ComplexityAnalyzer instance."""
    performance_config_path = "tests/assests/model_preformance.json"
    return ComplexityAnalyzer(performance_config_path=performance_config_path)

def test_calculate_complexity(complexity_analyzer):
    """Tests whether the complexity analysis returns a valid score."""
    prompt = "Write a Python function to compute Fibonacci numbers recursively."
    score = complexity_analyzer.calculate_complexity(prompt)
    assert 0 <= score <= 100

def test_get_complexity_details(complexity_analyzer):
    """Tests the complexity details breakdown."""
    prompt = "Write a SQL query to find top 5 customers."
    details = complexity_analyzer.get_complexity_details(prompt)
    
    assert "recommended_model" in details
    assert "overall_complexity" in details
    assert 0 <= details["overall_complexity"] <= 100
    assert details["token_count"] > 0
