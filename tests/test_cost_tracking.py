import pytest
from nadir.cost.tracker import CostTracker

def test_cost_tracker():
    tracker = CostTracker()
    tracker.add_cost("gpt-3.5-turbo", 1000, 500)

    cost_breakdown = tracker.get_cost_breakdown()
    assert cost_breakdown["total_cost"] > 0
    assert cost_breakdown["input_cost"] > 0
    assert cost_breakdown["output_cost"] > 0
