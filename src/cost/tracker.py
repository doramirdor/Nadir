from src import logging

class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.input_cost = 0.0
        self.output_cost = 0.0

    def add_cost(self, input_tokens: int, output_tokens: int, cost_per_1k_tokens_input: float, cost_per_1k_tokens_output: float) -> None:
        """
        Add the cost of a request to the total cost, tracking input and output costs separately.
        
        :param input_tokens: Number of input tokens used in the request.
        :param output_tokens: Number of output tokens generated in the response.
        :param cost_per_1k_tokens_input: Cost per 1,000 input tokens for the model.
        :param cost_per_1k_tokens_output: Cost per 1,000 output tokens for the model.
        """
        input_cost = (input_tokens / 1000) * cost_per_1k_tokens_input
        output_cost = (output_tokens / 1000) * cost_per_1k_tokens_output
        total_cost = input_cost + output_cost

        self.input_cost += input_cost
        self.output_cost += output_cost
        self.total_cost += total_cost

        logging.info(
            f"Added input cost: ${input_cost:.6f}, output cost: ${output_cost:.6f}, total: ${total_cost:.6f}. "
            f"Running total: input ${self.input_cost:.6f}, output ${self.output_cost:.6f}, total ${self.total_cost:.6f}"
        )

    def get_total_cost(self) -> float:
        """
        Get the total cost incurred so far.
        
        :return: Total cost in USD.
        """
        return self.total_cost

    def get_cost_breakdown(self) -> dict:
        """
        Get a breakdown of the costs into input, output, and total.
        
        :return: A dictionary containing the cost breakdown.
        """
        return {
            "input_cost": round(self.input_cost, 6),
            "output_cost": round(self.output_cost, 6),
            "total_cost": round(self.total_cost, 6)
        }
