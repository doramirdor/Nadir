import re
from typing import Dict, List
from functools import lru_cache

class CodeCompressor:
    """
    Auto-detecting code compressor supporting:
    Python, Java, JavaScript, TypeScript, JSX (React), SQL, Go
    """
    
    LANGUAGE_RULES = {
        'python': {
            'keywords': {'def', 'class', 'import', 'from', 'async', 'with', '#', 'lambda'},
            'compress_patterns': [
                (r'#.*', ''),  # Remove comments
                (r'(?<=\()\s+', ''),   # Remove space after '('
                (r'\s+(?=\))', ''),    # Remove space before ')'
                (r'\b(self|cls)\b', ''),  # Remove self/cls references
                (r'->.*?:', ':')       # Remove type hints on function returns
            ]
        },
        'java': {
            'keywords': {'class', 'public', 'void', 'new', 'package', 'System.out'},
            'compress_patterns': [
                (r'//.*', ''),         # Remove comments
                (r'@\w+\b', ''),       # Remove annotations
                (r'\b(final|protected)\b', ''),  # Remove modifiers
                (r'throws \w+', '')    # Remove throws statements
            ]
        },
        'javascript': {
            'keywords': {'function', 'const', 'export', '=>', 'require(', 'document.'},
            'compress_patterns': [
                (r'//.*', ''),         # Remove single-line comments
                (r'\s*([{}();,=])\s*', r'\1'),   # Remove extra space
                (r'function\s*\w*\s*\(', 'fn(')  # Shorten function declarations
            ]
        },
        'typescript': {
            'keywords': {'interface', 'type', 'enum', 'declare', ': string', ': number'},
            'compress_patterns': [
                (r':\s*\w+;', ';'),    # Simplify types
                (r'<\w+>', ''),        # Remove generic type params
                (r' as \w+', '')       # Remove 'as' type assertions
            ]
        },
        'react': {
            'keywords': {'useState', 'useEffect', '</', '/>', 'className='},
            'compress_patterns': [
                (r'className=', 'cls='),
                (r'\s+props\b', ''),
                (r'{\s*\w+\s*}', '{}'),
                (r'<([A-Z]\w+)\b', r'<\1')
            ]
        },
        'sql': {
            'keywords': {'SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'GROUP BY', 'UNION'},
            'compress_patterns': [
                (r'\bAS\s+\w+', ''),  # Remove aliases
                (r'\s+--.*', ''),     # Remove single-line comments
                (r'(?i)\bINNER\b', ''),  # Remove 'INNER'
                (r',\s*', ',')
            ]
        },
        'go': {
            'keywords': {'package main', 'func ', ':=', 'err :=', 'go '},
            'compress_patterns': [
                (r'\b(err error)\b', 'err'),
                (r'//.*', ''),
                (r'\bvar\b', ''),
                (r' := ', '=')
            ]
        }
    }

    def __init__(self, max_context_lines: int = 25, keep_imports: bool = True):
        """
        :param max_context_lines: Maximum lines to preserve in final output.
        :param keep_imports: Whether to preserve import/package lines in the first pass.
        """
        self.max_context_lines = max_context_lines
        self.keep_imports = keep_imports

    @lru_cache(maxsize=2000)
    def compress(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Compress code with automatic language detection and repeated passes if needed.
        """
        lang = self._detect_language(prompt)
        lang_rules = self.LANGUAGE_RULES[lang]
        
        compressed = self._initial_compress(prompt, lang_rules)
        # Repeat compression until within max_tokens
        while self._token_count(compressed) > max_tokens:
            compressed = self._aggressive_compress(compressed, lang_rules)
        return compressed

    def _detect_language(self, code: str) -> str:
        """Auto-detect programming language."""
        scores = {lang: 0 for lang in self.LANGUAGE_RULES}
        
        # Quick heuristics for certain markers
        if re.search(r'</|/>', code):
            return 'react'
        if 'package main' in code:
            return 'go'
        if re.search(r'SELECT|FROM|WHERE', code, re.IGNORECASE):
            return 'sql'
        
        # Score lines for other languages
        for line in code.split('\n'):
            for lang, rules in self.LANGUAGE_RULES.items():
                if any(kw in line for kw in rules['keywords']):
                    scores[lang] += 1
                    
        return max(scores, key=lambda k: scores[k])

    def _initial_compress(self, code: str, rules: Dict) -> str:
        """First-pass compression using language-specific patterns."""
        lines = self._preprocess_code(code, rules)
        compressed_lines = []
        
        for line in lines:
            cline = line
            # Apply language-specific patterns
            for pattern, repl in rules['compress_patterns']:
                cline = re.sub(pattern, repl, cline)
            # Optionally limit line length
            compressed_lines.append(cline[:120])
            
        # Truncate to max_context_lines
        return '\n'.join(compressed_lines[:self.max_context_lines])

    def _aggressive_compress(self, code: str, rules: Dict) -> str:
        """
        2nd pass compression:
          1) Remove line comments (#, //, --)
          2) Keep lines that have:
             - typical code structures (assignment, function calls)
             - known keywords from the language rule set
          3) Drop empty lines
          4) Return up to max_context_lines
        """
        # Remove line comments
        code_no_comments = re.sub(r'(#|//|--).*', '', code)
        
        # Keep lines with typical code elements
        important_lines = []
        for line in code_no_comments.split('\n'):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if (re.search(r'\w+\.\w+\(', line_stripped) or  # e.g., model.predict(
                re.search(r'\w+\s*=\s*\w+', line_stripped) or  # e.g., x = 5
                any(kw in line_stripped for kw in rules['keywords'])):
                important_lines.append(line_stripped)
        
        # Truncate
        return '\n'.join(important_lines[:self.max_context_lines])

    def _preprocess_code(self, code: str, rules: Dict) -> List[str]:
        """
        Clean and filter code lines for the first pass:
          - Remove empty lines
          - If keep_imports=True, keep lines starting with import/package or containing language keywords
          - Otherwise, keep up to max_context_lines
        """
        lines = [line for line in code.split('\n') if line.strip()]
        if not self.keep_imports:
            return lines[:self.max_context_lines]
        
        # keep_imports=True => preserve lines that either start with recognized import forms or have known keywords
        result = []
        for line in lines:
            line_stripped = line.strip()
            # Check standard import/package forms
            if re.match(r'^(import|package|require)', line_stripped):
                result.append(line_stripped)
            # Also keep lines that include known keywords
            elif any(kw in line_stripped for kw in rules['keywords']):
                result.append(line_stripped)
        return result

    def _token_count(self, text: str) -> int:
        """Approximate token count via word boundaries."""
        return len(re.findall(r'\b\w+\b', text))
