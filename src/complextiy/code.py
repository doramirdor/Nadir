import re
import ast
import math
import statistics
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import tiktoken

from src.complextiy import BaseComplexityAnalyzer


class CodeComplexityAnalyzer(BaseComplexityAnalyzer):
    """
    A complexity analyzer specialized for code and code completion.
    Focuses on Python, JavaScript (React), SQL, Java, Scala, HTML, and Go.
    """

    def __init__(self, tokenizer_name: str = "cl100k_base"):
        # Load the tokenizer (OpenAI's tiktoken, for example)
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

        # Language-specific keywords or constructs
        self.code_keywords = {
            'python': {
                'def', 'class', 'import', 'from', 'lambda', 'async', 'await',
                'try', 'except', 'raise'
            },
            'javascript': {
                'function', 'const', 'let', 'export', 'require',
                'import', '=>'
            },
            'react': {
                'React', 'useState', 'useEffect', 'useContext', 'useRef',
                'useReducer', 'useMemo', 'useCallback'
            },
            'sql': {
                'SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY',
                'CREATE TABLE', 'INSERT INTO', 'UPDATE', 'DELETE'
            },
            'java': {
                'public', 'static', 'void', 'class', 'extends', 'implements',
                'import', 'package'
            },
            'scala': {
                'object', 'def', 'val', 'var', 'trait', 'extends', 'lazy'
            },
            'go': {
                'func', 'package', 'import', 'struct', 'interface', 'go', 'chan'
            },
            'html': {
                '<html>', '<head>', '<body>', '<div', '</html>', '</head>',
                '</body>', '</div>'
            }
        }

        # Additional code structure patterns for deeper analysis
        self.control_flow_keywords = {
            'if', 'else', 'for', 'while', 'switch', 'case', 'when',
            'try', 'catch', 'except', 'finally'
        }

    def calculate_complexity(self, prompt: str) -> float:
        """
        Calculate an overall complexity score (0-100) for the code snippet.
        Involves token, structural, and domain complexity.
        """
        is_code, main_lang = self._detect_code_type(prompt)
        if not is_code:
            # If we don't detect it as code at all, treat as minimal code complexity
            return 0.0

        # Gather metrics
        token_count = len(self.tokenizer.encode(prompt))
        token_score = self._calculate_token_complexity(prompt)
        structural_score, _ = self._calculate_structural_complexity(
            prompt, main_lang, return_details=True
        )
        # Weighted combination
        # You can tune these weights based on your preference
        combined_score = (token_score * 0.6) + (structural_score * 0.4)

        # Optional: Adjust for extremely short or long code snippets
        if token_count < 20:
            combined_score *= (math.log(token_count + 1) / 4)
        elif token_count > 4000:  # if you consider 4k tokens "long"
            combined_score *= 1 + (token_count - 4000) / 8000

        # Ensure range is 0-100
        return min(max(combined_score, 0), 100)

    def get_complexity_details(self, prompt: str) -> Dict[str, Any]:
        """
        Return a dictionary with detailed complexity metrics.
        """
        is_code, main_lang = self._detect_code_type(prompt)
        if not is_code:
            # Non-code fallback
            return {
                "language_detected": None,
                "overall_complexity": 0.0,
                "token_count": len(self.tokenizer.encode(prompt)),
            }

        token_score = self._calculate_token_complexity(prompt)
        structural_score, structure_details = self._calculate_structural_complexity(
            prompt, main_lang, return_details=True
        )
        overall_score = self.calculate_complexity(prompt)

        return {
            "language_detected": main_lang,
            "overall_complexity": overall_score,
            "token_complexity": token_score,
            "structural_complexity": structural_score,
            **structure_details,  # e.g., function counts, classes, etc.
            "token_count": len(self.tokenizer.encode(prompt)),
        }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _detect_code_type(self, text: str) -> Tuple[bool, str]:
        """
        Scan for language-specific keywords. Return (is_code, detected_language).
        If multiple match, pick the one with the most hits.
        """
        scores = defaultdict(int)
        for lang, keywords in self.code_keywords.items():
            for kw in keywords:
                # Case-insensitive search
                if re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE):
                    scores[lang] += 1

        if not scores:
            return (False, "")

        # Pick the language with the highest keyword count
        main_lang = max(scores, key=scores.get)
        return (True, main_lang)

    def _calculate_token_complexity(self, text: str) -> float:
        """
        Score based on tokens (vocab diversity, average token length, total token count).
        """
        tokens = self.tokenizer.encode(text)
        if not tokens:
            return 0.0

        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        ttr = unique_tokens / total_tokens  # Type-Token Ratio

        # Decoded tokens to measure character length
        decoded = [self.tokenizer.decode([t]) for t in tokens]
        avg_length = sum(len(t) for t in decoded) / total_tokens

        # Heuristics-based scoring
        score = (ttr * 40) + (avg_length * 3) + (math.log(total_tokens + 1) * 7)

        return min(max(score, 0), 100)

    def _calculate_structural_complexity(
        self, text: str, lang: str, return_details: bool = False
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze code structure: functions, classes, control flow, AST depth (for Python).
        Return a complexity score and a dict with details if `return_details` is True.
        """
        details = {}

        # Basic structure metrics
        function_count = len(re.findall(r'\b(function|def)\b', text, re.IGNORECASE))
        class_count = len(re.findall(r'\bclass\b', text, re.IGNORECASE))
        control_flows = sum(
            len(re.findall(r'\b' + kw + r'\b', text, re.IGNORECASE))
            for kw in self.control_flow_keywords
        )
        nesting_depth = self._estimate_nesting_depth(text)

        # If Python, attempt AST parse to measure function nesting or complexity
        ast_depth = 0
        if lang.lower() == "python":
            try:
                tree = ast.parse(text)
                ast_depth = self._max_ast_depth(tree)
            except Exception:
                pass

        # Combine structure
        base_score = (
            function_count * 4
            + class_count * 6
            + control_flows * 3
            + nesting_depth * 2
            + ast_depth * 4
        )
        final_score = min(base_score, 100)

        # If user wants details, prepare them
        if return_details:
            details.update(
                {
                    "function_count": function_count,
                    "class_count": class_count,
                    "control_flow_count": control_flows,
                    "nesting_depth": nesting_depth,
                    "ast_depth": ast_depth,
                }
            )
            return final_score, details

        return final_score, {}

    def _estimate_nesting_depth(self, text: str) -> int:
        """
        Estimate nesting by indentation (heuristic).
        """
        max_depth = 0
        for line in text.split('\n'):
            # Count leading spaces or tabs
            leading_spaces = len(line) - len(line.lstrip(' '))
            leading_tabs = len(line) - len(line.lstrip('\t'))
            indent_level = max(leading_spaces / 4, leading_tabs)
            if indent_level > max_depth:
                max_depth = indent_level
        return int(max_depth)

    def _max_ast_depth(self, node) -> int:
        """
        Recursively compute the maximum depth of an AST (Python only).
        """
        if isinstance(node, ast.AST):
            return 1 + max((self._max_ast_depth(child) for child in ast.iter_child_nodes(node)), default=0)
        return 0
