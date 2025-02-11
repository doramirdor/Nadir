from litellm import cost_per_token
from nadir import logging


class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.input_cost = 0.0
        self.output_cost = 0.0

    def add_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Add the cost of a request to the total cost, tracking input and output costs separately.

        :param model: The model name (e.g., "gpt-3.5-turbo").
        :param prompt_tokens: Number of input tokens used in the request.
        :param completion_tokens: Number of output tokens generated in the response.
        """
        # Calculate costs using litellm's `cost_per_token`
        prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )

        # Update costs
        self.input_cost += prompt_tokens_cost_usd_dollar
        self.output_cost += completion_tokens_cost_usd_dollar
        self.total_cost += prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar

        logging.info(
            f"Added input cost: ${prompt_tokens_cost_usd_dollar:.6f}, output cost: ${completion_tokens_cost_usd_dollar:.6f}, "
            f"total: ${prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar:.6f}. "
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