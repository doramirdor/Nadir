import re
import math
import statistics
import tiktoken
from typing import Dict, Any, List

class ComplexityAnalyzer:
    def __init__(self, tokenizer_name: str = "cl100k_base"):
        """
        Initialize the complexity analyzer with a specified tokenizer.
        
        :param tokenizer_name: Tiktoken encoding name.
        """
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
    def calculate_complexity(self, prompt: str) -> float:
        """
        Compute an overall complexity score (0-100) based on multiple factors:
        - Token complexity (lexical diversity, average token length, etc.)
        - Linguistic complexity (readability, technical terms, etc.)
        - Structural complexity (variability in sentence/paragraph structure, etc.)
        
        :param prompt: Input text to analyze.
        :return: A single numeric complexity score between 0 and 100.
        """
        # Calculate all sub-metrics
        token_score = self._calculate_token_complexity(prompt)
        linguistic_score = self._calculate_linguistic_complexity(prompt)
        structural_score = self._calculate_structural_complexity(prompt)

        # Weighted combination of the three metrics
        weights = {
            'token': 0.4,
            'linguistic': 0.4,
            'structural': 0.2
        }
        complexity = (
            token_score * weights['token'] +
            linguistic_score * weights['linguistic'] +
            structural_score * weights['structural']
        )

        # Optional: Additional dampening for extremely short prompts
        token_count = len(self.tokenizer.encode(prompt))
        if token_count < 10:
            # Scale down linearly as token_count goes from 1..9
            complexity *= (token_count / 10.0)

        # Ensure the final score is within 0-100
        return min(max(complexity, 0), 100)

    def get_complexity_details(self, prompt: str) -> Dict[str, Any]:
        """
        Provide a detailed breakdown of the complexity metrics.

        :param prompt: Input text to analyze.
        :return: A dictionary containing each sub-metric and the overall complexity.
        """
        token_score = self._calculate_token_complexity(prompt)
        linguistic_score = self._calculate_linguistic_complexity(prompt)
        structural_score = self._calculate_structural_complexity(prompt)
        overall_complexity = self.calculate_complexity(prompt)

        return {
            'overall_complexity': overall_complexity,
            'token_complexity': token_score,
            'linguistic_complexity': linguistic_score,
            'structural_complexity': structural_score,
            'token_count': len(self.tokenizer.encode(prompt))
        }
    
    def _calculate_token_complexity(self, text: str) -> float:
        """
        Calculate complexity based on lexical and token characteristics:
          - Type-token ratio (unique vs. total tokens).
          - Average token length.
          - Total token count (log-scaled).
        
        The combined score is then scaled to a 0-100 range.
        
        :param text: Input text.
        :return: A float representing token complexity.
        """
        tokens = self.tokenizer.encode(text)
        if not tokens:
            return 0.0

        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        # Type-token ratio (higher means more diverse vocabulary)
        ttr = unique_tokens / total_tokens

        # ---- ADJUSTMENT FOR VERY SHORT TEXTS ----
        # If there are fewer than 20 tokens, reduce the TTR influence.
        if total_tokens < 20:
            ttr *= (total_tokens / 20.0)

        # Approximate average token length
        decoded_tokens = [self.tokenizer.decode([tok]) for tok in tokens]
        avg_token_length = sum(len(dt) for dt in decoded_tokens) / total_tokens
        
        # Combine measures:
        #   - TTR scaled up to ~60
        #   - Average token length up to ~30
        #   - Log-scaled total tokens up to some moderate range
        score = (
            (ttr * 60) +
            (avg_token_length * 3) +
            (math.log2(total_tokens + 1) * 5)
        )
        
        return min(max(score, 0), 100)

    def _calculate_linguistic_complexity(self, text: str) -> float:
        """
        Analyze linguistic complexity by:
         - Flesch Reading Ease Score (inverted to measure complexity).
         - Simple detection of technical terms (CamelCase or multiple uppercase letters).
         - Weighted combination of the two to yield a 0-100 scale.
        
        :param text: Input text.
        :return: A float for linguistic complexity.
        """
        if not text.strip():
            return 0.0
        
        reading_ease = self._flesch_reading_ease(text)
        # Invert and clamp the reading ease result
        reading_ease_clamped = max(min(reading_ease, 100), -50)
        # Shift negative floor, invert, clamp
        flesch_complexity = 100 - (reading_ease_clamped + 50)
        flesch_complexity = min(max(flesch_complexity, 0), 100)

        # Technical term detection (e.g., "NeuralNetwork", "HTTPResponse", etc.)
        technical_terms = len(re.findall(r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b", text))
        # Scale up to 50 max
        tech_factor = min(technical_terms * 2, 50)

        # Weighted combination
        combined_score = 0.7 * flesch_complexity + 0.3 * tech_factor
        return min(max(combined_score, 0), 100)

    def _calculate_structural_complexity(self, text: str) -> float:
        """
        Assess structural complexity by:
         - Sentence length variability (stdev of words per sentence).
         - Paragraph count factor.
         - Weighted combination scaled to 0-100.
        
        :param text: Input text.
        :return: A float for structural complexity.
        """
        # Split into sentences
        raw_sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Calculate words per sentence
        wps = [len(s.split()) for s in sentences]
        
        # If only 1 sentence, stdev = 0
        if len(wps) == 1:
            stdev_sentences = 0
        else:
            stdev_sentences = statistics.pstdev(wps)

        # Paragraph count by double newline or your own logic
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)

        structural_score = (stdev_sentences * 3) + (paragraph_count * 2)
        return min(max(structural_score, 0), 100)
    
    def _flesch_reading_ease(self, text: str) -> float:
        """
        Rough calculation of the Flesch Reading Ease Score:
            FRE = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words).
        
        Very naive approach for syllable counting. For more accurate results, 
        consider a library like 'textstat'.
        
        :param text: Input text.
        :return: Flesch Reading Ease score (higher = simpler).
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)
        
        words = re.findall(r'\w+', text)
        word_count = max(len(words), 1)
        
        # Very simple syllable heuristic: count vowel groups
        syllable_count = sum(len(re.findall(r'[aeiouyAEIOUY]+', w)) for w in words)
        
        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_count / word_count
        
        reading_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        return reading_ease
