import os
import re
import ast
import math
import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from src.nadir.complexity import BaseComplexityAnalyzer

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CodeComplexityAnalyzer(BaseComplexityAnalyzer):
    """
    A production-ready static code complexity analyzer.
    
    It evaluates the complexity of source code using static analysis techniques
    without relying on external LLMs. It provides a fine-grained complexity score
    and selects the best LLM model based on performance, speed, and cost.
    """

    def __init__(
        self,
        candidate_names: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        performance_config_path: Optional[str] = None
    ):
        """
        Initialize the analyzer and load available model candidates.

        :param candidate_names: Optional list of candidate unique names.
        :param providers: Optional list of providers (e.g., "OpenAI", "Anthropic").
        :param performance_config_path: Path to the model performance JSON file.
        """
        super().__init__(None, candidate_names, providers, performance_config_path)
        logging.info(f"Initialized CodeComplexityAnalyzer with {len(self.candidate_models)} candidate(s).")

        # Initialize language patterns
        self._init_language_patterns()

    def _init_language_patterns(self):
        """Define language-specific complexity markers and keywords"""
        self.language_patterns = {
            'python': {
                'keywords': {'def', 'class', 'import', 'from', 'async', 'await'},
                'complexity_markers': {
                    'generators': r'yield\b',
                    'decorators': r'@\w+',
                    'lambda': r'lambda\b',
                    'list_comprehension': r'\[.*for.*in.*\]'
                }
            },
            'javascript': {
                'keywords': {'function', 'class', 'const', 'let', 'async', 'await'},
                'complexity_markers': {
                    'promises': r'new Promise|\.then|\.catch',
                    'callbacks': r'function.*\(.*\).*{.*}',
                    'arrow_functions': r'=>'
                }
            },
            'sql': {
                'keywords': {'SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY'},
                'complexity_markers': {
                    'joins': r'JOIN\b',
                    'subqueries': r'\(\s*SELECT',
                    'window_functions': r'OVER\s*\('
                }
            }
        }

    def calculate_complexity(self, code: str) -> float:
        """
        Compute an overall complexity score (0-100) for the given code snippet.

        :param code: Input code to analyze.
        :return: Complexity score (0-100).
        """
        details = self.get_complexity_details(code)
        complexity_score = details.get("overall_complexity", 0.0)
        logging.info(f"Calculated complexity: {complexity_score}")
        return complexity_score

    def get_complexity_details(self, code: str) -> Dict[str, Any]:
        """
        Perform an in-depth complexity analysis of the given code.

        :param code: The source code snippet to analyze.
        :return: Dictionary containing complexity metrics.
        """
        is_code, lang = self._detect_code_type(code)
        if not is_code:
            return {
                "language_detected": None,
                "overall_complexity": 0.0,
                "recommended_model": "N/A",
                "token_count": len(re.findall(r'\S+', code))
            }

        # Token, structural, and language-specific analysis
        token_metrics = self._analyze_token_metrics(code)
        structural_metrics = self._analyze_structural_metrics(code, lang)
        language_specific_metrics = self._analyze_language_specific(code, lang)

        # Compute final complexity score
        overall_complexity = self._calculate_overall_complexity(
            token_metrics, structural_metrics, language_specific_metrics
        )

        # **Select the Best Model**
        recommended_model = self._select_best_model(overall_complexity)

        return {
            "language_detected": lang,
            "overall_complexity": overall_complexity,
            "recommended_model": recommended_model,
            "token_metrics": token_metrics,
            "structural_metrics": structural_metrics,
            "language_specific_metrics": language_specific_metrics,
            "token_count": len(re.findall(r'\S+', code))
        }

    def _detect_code_type(self, text: str) -> Tuple[bool, str]:
        """
        Detect the programming language of a given code snippet.
        """
        scores = defaultdict(int)
        for lang, patterns in self.language_patterns.items():
            for kw in patterns['keywords']:
                if re.search(r'\b' + re.escape(kw) + r'\b', text):
                    scores[lang] += 1

        if not scores:
            return False, ""
        return True, max(scores, key=scores.get)

    def _select_best_model(self, complexity_score: float) -> str:
        """
        Selects the best model based on complexity score, performance, and cost.
        """
        if not self.candidate_models:
            return "N/A"

        # Prioritize performance for high complexity, speed for low complexity
        sorted_candidates = sorted(
            self.candidate_models,
            key=lambda c: (
                abs(complexity_score - float(c.get("Quality Index", 50))),  # Closer Quality Index is better
                -float(c.get("MedianTokens/s", 1)),  # Faster speed preferred
                float(c.get("Blended Price (USD per 1M tokens)", 100))  # Lower cost preferred
            )
        )

        return sorted_candidates[0]["unique_name"] if sorted_candidates else "N/A"

    def _analyze_token_metrics(self, code: str) -> Dict[str, Any]:
        """Token-based complexity analysis."""
        tokens = re.findall(r'\S+', code)
        unique_tokens = set(tokens)

        return {
            "token_count": len(tokens),
            "unique_tokens": len(unique_tokens),
            "token_diversity": len(unique_tokens) / len(tokens) if tokens else 0,
            "average_token_length": sum(len(t) for t in tokens) / len(tokens) if tokens else 0
        }

    def _analyze_structural_metrics(self, code: str, lang: str) -> Dict[str, Any]:
        """Analyze the structure of the code snippet."""
        lines = code.splitlines()
        return {
            "line_count": len(lines),
            "empty_lines": sum(1 for line in lines if not line.strip()),
            "nesting_depth": self._estimate_nesting_depth(code)
        }

    def _estimate_nesting_depth(self, code: str) -> int:
        """Estimate the maximum nesting depth based on indentation."""
        max_depth = 0
        for line in code.splitlines():
            indent_level = len(line) - len(line.lstrip(' '))
            max_depth = max(max_depth, indent_level // 4)
        return max_depth

    def _analyze_language_specific(self, code: str, lang: str) -> Dict[str, Any]:
        """Perform language-specific complexity analysis."""
        if lang not in self.language_patterns:
            return {}

        patterns = self.language_patterns[lang]
        return {
            "keyword_density": sum(len(re.findall(r'\b' + kw + r'\b', code))
                                   for kw in patterns['keywords']) / len(code.splitlines()),
            "complexity_markers": {name: len(re.findall(pattern, code))
                                   for name, pattern in patterns['complexity_markers'].items()}
        }
