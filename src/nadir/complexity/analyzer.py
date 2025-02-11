import os
import re
import math
import statistics
import logging
import tiktoken
from typing import Dict, Any, List, Optional
from src.nadir.complexity import BaseComplexityAnalyzer
from src.nadir.utils import load_performance_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ComplexityAnalyzer(BaseComplexityAnalyzer):
    """
    A text complexity analyzer that evaluates prompts based on:
    - Token complexity (lexical diversity, length, density)
    - Linguistic complexity (readability, technical terms)
    - Structural complexity (sentence variability, paragraph count)
    
    The analyzer also selects the best model for processing based on performance, speed, and cost.
    """

    def __init__(
        self,
        tokenizer_name: str = "cl100k_base",
        candidate_names: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        performance_config_path: Optional[str] = None
    ):
        """
        Initialize the analyzer with a tokenizer and load model candidates.

        :param tokenizer_name: Tiktoken encoding name.
        :param candidate_names: Optional list of candidate unique names.
        :param providers: Optional list of providers (e.g., "OpenAI", "Anthropic").
        :param performance_config_path: Path to the model performance JSON file.
        """
        super().__init__(None, candidate_names, providers, performance_config_path)
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        logging.info(f"Initialized ComplexityAnalyzer with {len(self.candidate_models)} candidate(s).")

    def calculate_complexity(self, prompt: str) -> float:
        """
        Compute an overall complexity score (0-100) based on multiple factors:
        - Token complexity (lexical diversity, average token length, etc.)
        - Linguistic complexity (readability, technical terms, etc.)
        - Structural complexity (variability in sentence/paragraph structure, etc.)
        
        :param prompt: Input text to analyze.
        :return: A numeric complexity score between 0 and 100.
        """
        details = self.get_complexity_details(prompt)
        complexity_score = details.get("overall_complexity", 0.0)
        logging.info(f"Calculated complexity: {complexity_score}")
        return complexity_score

    def get_complexity_details(self, prompt: str) -> Dict[str, Any]:
        """
        Provide a detailed breakdown of the complexity metrics and select the best model.

        :param prompt: Input text to analyze.
        :return: A dictionary containing sub-metrics and the overall complexity score.
        """
        token_score = self._calculate_token_complexity(prompt)
        linguistic_score = self._calculate_linguistic_complexity(prompt)
        structural_score = self._calculate_structural_complexity(prompt)

        overall_complexity = (
            (token_score * 0.4) +
            (linguistic_score * 0.4) +
            (structural_score * 0.2)
        )

        recommended_model = self._select_best_model(overall_complexity)

        return {
            "overall_complexity": min(max(overall_complexity, 0), 100),
            "recommended_model": recommended_model,
            "token_complexity": token_score,
            "linguistic_complexity": linguistic_score,
            "structural_complexity": structural_score,
            "token_count": len(self.tokenizer.encode(prompt))
        }

    def _select_best_model(self, complexity_score: float) -> str:
        """
        Selects the best model based on complexity score, performance, and cost.
        """
        if not self.candidate_models:
            return "N/A"

        sorted_candidates = sorted(
            self.candidate_models,
            key=lambda c: (
                abs(complexity_score - float(c.get("Quality Index", 50))),  # Closest Quality Index
                -float(c.get("MedianTokens/s", 1)),  # Faster models preferred
                float(c.get("Blended Price (USD per 1M tokens)", 100))  # Lower cost preferred
            )
        )

        return sorted_candidates[0]["unique_name"] if sorted_candidates else "N/A"

    def _calculate_token_complexity(self, text: str) -> float:
        """
        Evaluate token complexity based on:
          - Type-token ratio (unique vs. total tokens).
          - Average token length.
          - Log-scaled total token count.
        """
        tokens = self.tokenizer.encode(text)
        if not tokens:
            return 0.0

        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        ttr = unique_tokens / total_tokens

        if total_tokens < 20:
            ttr *= (total_tokens / 20.0)

        decoded_tokens = [self.tokenizer.decode([tok]) for tok in tokens]
        avg_token_length = sum(len(dt) for dt in decoded_tokens) / total_tokens

        score = (
            (ttr * 60) +
            (avg_token_length * 3) +
            (math.log2(total_tokens + 1) * 5)
        )
        return min(max(score, 0), 100)

    def _calculate_linguistic_complexity(self, text: str) -> float:
        """
        Assess linguistic complexity using:
        - Inverted Flesch Reading Ease Score.
        - Technical term detection (CamelCase, uppercase words).
        """
        if not text.strip():
            return 0.0

        reading_ease = self._flesch_reading_ease(text)
        reading_ease_clamped = max(min(reading_ease, 100), -50)
        flesch_complexity = 100 - (reading_ease_clamped + 50)
        flesch_complexity = min(max(flesch_complexity, 0), 100)

        technical_terms = len(re.findall(r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b", text))
        tech_factor = min(technical_terms * 2, 50)

        return min(max((0.7 * flesch_complexity) + (0.3 * tech_factor), 0), 100)

    def _calculate_structural_complexity(self, text: str) -> float:
        """
        Analyze structural complexity based on:
        - Sentence length variability.
        - Number of paragraphs.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences:
            return 0.0

        wps = [len(s.split()) for s in sentences]
        stdev_sentences = statistics.pstdev(wps) if len(wps) > 1 else 0

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)

        return min(max((stdev_sentences * 3) + (paragraph_count * 2), 0), 100)

    def _flesch_reading_ease(self, text: str) -> float:
        """
        Roughly calculate the Flesch Reading Ease Score.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = max(len(sentences), 1)

        words = re.findall(r'\w+', text)
        word_count = max(len(words), 1)

        syllable_count = sum(len(re.findall(r'[aeiouyAEIOUY]+', w)) for w in words)

        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_count / word_count

        return 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
