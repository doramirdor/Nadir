import re
import math
import ast
import tiktoken
import statistics 
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from src.complextiy import BaseComplexityAnalyzer

class AdvancedComplexityAnalyzer(BaseComplexityAnalyzer):
    def __init__(self, tokenizer_name: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.code_keywords = {
            'python': {'def', 'class', 'import', 'from', 'try', 'except', 'raise'},
            'js': {'function', 'const', 'let', 'export', 'import'},
            'sql': {'SELECT', 'FROM', 'WHERE', 'JOIN', 'CREATE TABLE'},
            'generic': {'if', 'else', 'for', 'while', 'return', 'async', 'await'}
        }
        self.app_domain_terms = {
            'web': {'API', 'endpoint', 'middleware', 'SSR', 'CSR', 'hydration'},
            'database': {'schema', 'migration', 'index', 'ORM', 'ACID'},
            'auth': {'OAuth', 'JWT', 'session', 'cookie', 'SSO'},
            'devops': {'CI/CD', 'pipeline', 'container', 'orchestration'}
        }

    def calculate_complexity(self, token_complexity: float, linguistic_complexity: float, structural_complexity: float, token_count: int, is_code: bool) -> float:
        if is_code:
            weights = {
                'token': 0.25,
                'linguistic': 0.35,
                'structural': 0.4
            }
        else:
            weights = {
                'token': 0.3,
                'linguistic': 0.4,
                'structural': 0.3
            }

        complexity = (
            token_complexity * weights['token'] +
            linguistic_complexity * weights['linguistic'] +
            structural_complexity * weights['structural']
        )

        # Length normalization
        if token_count < 50:
            complexity *= math.log(token_count + 1) / 4
        elif token_count > 2000:
            complexity *= 1 + (token_count - 2000) / 5000

        return min(max(complexity, 0), 100)

    def get_complexity_details(self, prompt: str) -> Dict[str, Any]:
        is_code, lang = self._detect_code_type(prompt)
        code_metrics = self._analyze_code_structure(prompt, lang) if is_code else {}

        token_complexity = self._calculate_token_complexity(prompt, is_code)
        linguistic_complexity = self._calculate_linguistic_complexity(prompt, is_code)
        structural_complexity = self._calculate_structural_complexity(prompt, is_code, lang)
        token_count = len(self.tokenizer.encode(prompt))

        overall_complexity = self.calculate_complexity(
            token_complexity, linguistic_complexity, structural_complexity, token_count, is_code
        )

        return {
            'overall_complexity': overall_complexity,
            'token_complexity': token_complexity,
            'linguistic_complexity': linguistic_complexity,
            'structural_complexity': structural_complexity,
            'token_count': token_count,
            'code_type': lang if is_code else None,
            **code_metrics
        }

    def _calculate_token_complexity(self, text: str, is_code: bool) -> float:
        tokens = self.tokenizer.encode(text)
        if not tokens:
            return 0.0

        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        ttr = unique_tokens / total_tokens

        decoded = [self.tokenizer.decode([t]) for t in tokens]
        avg_length = sum(len(t) for t in decoded) / total_tokens

        if is_code:
            score = (ttr * 40) + (avg_length * 6) + (math.log(total_tokens + 1) * 8)
        else:
            score = (ttr * 50) + (avg_length * 4) + (math.log(total_tokens + 1) * 6)

        return min(max(score, 0), 100)

    def _calculate_linguistic_complexity(self, text: str, is_code: bool) -> float:
        if is_code:
            tech_terms = self._count_technical_terms(text)
            lang_features = self._count_language_specific_features(text)
            return min((tech_terms * 2) + (lang_features * 3), 100)

        reading_ease = self._flesch_reading_ease(text)
        tech_terms = self._count_domain_terms(text)
        return min((100 - reading_ease) * 0.7 + tech_terms * 0.3, 100)

    def _calculate_structural_complexity(self, text: str, is_code: bool, lang: str) -> float:
        if is_code:
            return self._calculate_code_structural_complexity(text, lang)
        return self._calculate_text_structural_complexity(text)

    def _detect_code_type(self, text: str) -> Tuple[bool, str]:
        lang_scores = defaultdict(int)
        for lang, keywords in self.code_keywords.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', text):
                    lang_scores[lang] += 1

        if lang_scores:
            main_lang = max(lang_scores, key=lang_scores.get)
            return (True, main_lang if main_lang != 'generic' else 'unknown')
        return (False, 'text')

    def _count_technical_terms(self, text: str) -> int:
        count = 0
        for terms in self.app_domain_terms.values():
            for term in terms:
                count += len(re.findall(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
        return count

    def _count_language_specific_features(self, text: str) -> int:
        features = {
            'python': ['@decorator', 'lambda ', ' listcomp', 'dictcomp'],
            'js': ['=>', 'Promise', 'useState', 'useEffect'],
            'sql': ['INDEXED BY', 'WITH RECURSIVE', 'OVER()']
        }
        count = 0
        for _, patterns in features.items():
            count += sum(len(re.findall(p, text)) for p in patterns)
        return count

    def _analyze_code_structure(self, text: str, lang: str) -> Dict[str, Any]:
        """
        Analyze the structure of the given text, extracting metrics like function count, class count,
        control flow count, nesting depth, and AST depth (if applicable).
        """
        metrics = {
            'functions': len(re.findall(r'\b(def|function)\b', text, re.IGNORECASE)),
            'classes': len(re.findall(r'\bclass\b', text, re.IGNORECASE)),
            'control_flow': len(re.findall(r'\b(if|else|for|while|switch|case)\b', text, re.IGNORECASE)),
            'nesting_depth': self._estimate_nesting_depth(text)
        }

        # Only attempt AST parsing for Python code
        if lang.lower() == "python":
            try:
                tree = ast.parse(text)
                metrics['ast_depth'] = self._max_ast_depth(tree)
            except (RecursionError, SyntaxError):
                metrics['ast_depth'] = 0
        else:
            metrics['ast_depth'] = 0  # Default for non-Python languages

        return metrics

    def _calculate_code_structural_complexity(self, text: str, lang: str) -> float:
        metrics = self._analyze_code_structure(text, lang)
        score = (
            metrics['functions'] * 5 +
            metrics['classes'] * 8 +
            metrics['control_flow'] * 3 +
            metrics.get('ast_depth', 0) * 4 +
            metrics['nesting_depth'] * 2
        )
        return min(score, 100)

    def _calculate_text_structural_complexity(self, text: str) -> float:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if len(sentences) < 2:
            return 0

        wps = [len(s.split()) for s in sentences]
        stdev = statistics.pstdev(wps) if len(wps) > 1 else 0
        return min((stdev * 2) + (len(paragraphs) * 1.5), 100)

    def _estimate_nesting_depth(self, text: str) -> int:
        max_depth = 0
        current_depth = 0
        for line in text.split('\n'):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            current_depth = indent // 4  # Assuming 4-space indentation
            max_depth = max(max_depth, current_depth)
        return max_depth

    def _max_ast_depth(self, node) -> int:
        if isinstance(node, ast.AST):
            return 1 + max((self._max_ast_depth(child) for child in ast.iter_child_nodes(node)), default=0)
        return 0

    # Keep existing _flesch_reading_ease implementation from previous analyzer
