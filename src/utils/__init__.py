import json
import logging
from typing import Dict, Any, Optional, List


def parse_cost(cost_str: Optional[str]) -> float:
    """
    Parses a cost string in the format '$XX.XX' and returns the cost per token.
    """
    if not cost_str:
        return 0.0
    try:
        # Extract numeric value from the string (removing '$' if present)
        cost_per_million = float(cost_str.replace("$", "").strip())
        # Convert to cost per token
        return cost_per_million / 1_000_000
    except (ValueError, IndexError):
        return 0.0


def safe_float(value, default=50.0):
    """
    Safely converts a value to float. If the value is None or invalid, returns the default.
    """
    try:
        return float(value.strip("%")) if value is not None else default
    except ValueError:
        return default


def load_performance_config(json_path: str) -> List[Dict[str, Any]]:
    """
    Loads performance configuration data from a JSON file and returns a list of candidate model dictionaries.

    Each candidate dictionary contains:
        - Performance Metrics:
            - "Quality Index"
            - "Chatbot Arena"
            - "MMLU"
            - "GPQA"
            - "MATH-500"
            - "HumanEval"
        - Pricing details:
            - "Blended Price (USD per 1M tokens)"
            - "Input Price (USD per 1M tokens)"
            - "Output Price (USD per 1M tokens)"
            - "Average Cost per Token (USD)"
        - Running Time Metrics:
            - "MedianTokens/s"
            - "P5Tokens/s"
            - "P25Tokens/s"
            - "P75Tokens/s"
            - "P95Tokens/s"
            - "Median First Chunk (s)"
            - "P5 First Chunk (s)"
            - "P25 First Chunk (s)"
            - "P75 First Chunk (s)"
            - "P95 First Chunk (s)"
    
    The full raw record is also stored under the key "raw" if needed.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    candidates = []
    for item in data:
        # Compute the unique name based on API Provider and Model.
        api_provider = item.get("route", "")
        model = item.get("API ID", "").strip()
        unique_name = f"{api_provider}/{model}"

        # Extract and normalize pricing
        blended_price = parse_cost(item.get("BlendedUSD/1M Tokens"))
        input_price = parse_cost(item.get("Input PriceUSD/1M Tokens"))
        output_price = parse_cost(item.get("Output PriceUSD/1M Tokens"))

        # Compute a simple average token cost (adjust weighting if needed)
        avg_token_cost = (input_price + output_price) / 2 if input_price and output_price else blended_price

        # Extract latency/running time details
        median_tokens_per_sec = safe_float(item.get("MedianTokens/s"), default=0.0)
        p5_tokens_per_sec = safe_float(item.get("P5Tokens/s"), default=0.0)
        p25_tokens_per_sec = safe_float(item.get("P25Tokens/s"), default=0.0)
        p75_tokens_per_sec = safe_float(item.get("P75Tokens/s"), default=0.0)
        p95_tokens_per_sec = safe_float(item.get("P95Tokens/s"), default=0.0)

        median_first_chunk = safe_float(item.get("MedianFirst Chunk (s)"), default=0.0)
        p5_first_chunk = safe_float(item.get("P5First Chunk (s)"), default=0.0)
        p25_first_chunk = safe_float(item.get("P25First Chunk (s)"), default=0.0)
        p75_first_chunk = safe_float(item.get("P75First Chunk (s)"), default=0.0)
        p95_first_chunk = safe_float(item.get("P95First Chunk (s)"), default=0.0)

        candidate = {
            "unique_name": unique_name,
            "api_provider": api_provider,
            "model": model,
            "Quality Index": safe_float(item.get("Quality Index")),
            "Chatbot Arena": safe_float(item.get("Chatbot Arena")),
            "MMLU": safe_float(item.get("MMLU")),
            "GPQA": safe_float(item.get("GPQA")),
            "MATH-500": safe_float(item.get("MATH-500")),
            "HumanEval": safe_float(item.get("HumanEval")),
            # Pricing details
            "Blended Price (USD per 1M tokens)": blended_price,
            "Input Price (USD per 1M tokens)": input_price,
            "Output Price (USD per 1M tokens)": output_price,
            "Average Cost per Token (USD)": avg_token_cost,
            # Running Time Metrics
            "MedianTokens/s": median_tokens_per_sec,
            "P5Tokens/s": p5_tokens_per_sec,
            "P25Tokens/s": p25_tokens_per_sec,
            "P75Tokens/s": p75_tokens_per_sec,
            "P95Tokens/s": p95_tokens_per_sec,
            "Median First Chunk (s)": median_first_chunk,
            "P5 First Chunk (s)": p5_first_chunk,
            "P25 First Chunk (s)": p25_first_chunk,
            "P75 First Chunk (s)": p75_first_chunk,
            "P95 First Chunk (s)": p95_first_chunk,
            "raw": item  # Store the raw record if needed
        }
        candidates.append(candidate)

    return candidates
