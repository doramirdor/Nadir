import re
import hashlib
from functools import lru_cache
from typing import List, Optional
from transformers import pipeline
from collections import Counter

class FasCompressor:
    """
    Cost-optimized prompt compressor using multiple non-LLM techniques:
    1. Intelligent truncation with position-aware importance scoring
    2. Keyphrase extraction using specialized models
    3. Redundancy removal
    4. Semantic preserving compression
    """
    
    def __init__(
        self,
        hf_keyphrase_model: str = "ml6team/keyphrase-extraction-kbir-inspec",
        cache_size: int = 1000,
        min_quality: float = 0.4
    ):
        self.keyphrase_extractor = pipeline("token-classification", model=hf_keyphrase_model)
        self.min_quality = min_quality
        self.cache_size = cache_size

    @lru_cache(maxsize=None)
    def compress(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Smart compression pipeline with accuracy preservation
        """
        original_hash = self._content_hash(prompt)
        
        # Compression pipeline
        compressed = prompt
        for method in [
            self._remove_redundancies,
            self._smart_truncate,
            self._extract_keyphrases,
            self._position_aware_compress
        ]:
            compressed = method(compressed, max_tokens)
            if self._token_count(compressed) <= max_tokens:
                break
                
        # Final check against original content
        if self._content_hash(compressed) == original_hash:
            return self._force_truncate(compressed, max_tokens)
            
        return compressed

    def _token_count(self, text: str) -> int:
        """Fast approximate token counting"""
        return len(text.split())

    def _content_hash(self, text: str) -> str:
        """Create content fingerprint"""
        return hashlib.md5(text.encode()).hexdigest()

    def _remove_redundancies(self, text: str, max_tokens: int) -> str:
        """
        Remove duplicate phrases and repeated patterns while preserving meaning
        """
        sentences = re.split(r'[.!?]+\s*', text)
        unique_sentences = []
        seen = set()
        
        for sent in sentences:
            sent_hash = self._content_hash(sent)
            if sent_hash not in seen:
                seen.add(sent_hash)
                unique_sentences.append(sent)
                
        compressed = '. '.join(unique_sentences)
        return self._force_truncate(compressed, max_tokens)

    def _smart_truncate(self, text: str, max_tokens: int) -> str:
        """
        Position-aware truncation that prioritizes important sections
        """
        words = text.split()
        if len(words) <= max_tokens:
            return text
            
        # Score words by position and frequency
        word_scores = {}
        mid_point = len(words) // 2
        word_counts = Counter(words)
        
        for idx, word in enumerate(words):
            position_score = 1 - abs(idx - mid_point)/len(words)
            frequency_score = word_counts[word]/len(words)
            word_scores[word] = position_score + frequency_score
            
        # Select top words maintaining order
        sorted_words = sorted(enumerate(words), key=lambda x: -word_scores[x[1]])
        keep_indices = set([i for i, _ in sorted_words[:max_tokens]])
        
        return ' '.join([word for i, word in enumerate(words) if i in keep_indices])

    def _extract_keyphrases(self, text: str, max_tokens: int) -> str:
        """
        Extract keyphrases using specialized model with quality threshold
        """
        try:
            keyphrases = self.keyphrase_extractor(text)
            filtered = [
                kp["word"] for kp in keyphrases 
                if kp["score"] > self.min_quality
            ]
            if filtered:
                return ' '.join(filtered)
        except:
            pass
            
        return self._force_truncate(text, max_tokens)

    def _position_aware_compress(self, text: str, max_tokens: int) -> str:
        """
        Combine sentence importance scoring with term frequency
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentence_scores = []
        
        # Score sentences by position and term frequency
        max_pos = len(sentences)
        word_counts = Counter(text.split())
        
        for idx, sent in enumerate(sentences):
            pos_score = 1 - (idx/max_pos)  # Favor earlier sentences
            term_score = sum(word_counts[word] for word in sent.split())/len(sent.split())
            sentence_scores.append((pos_score + term_score, sent))
            
        # Select top sentences
        sorted_sentences = sorted(sentence_scores, key=lambda x: -x[0])
        selected = []
        total_tokens = 0
        
        for score, sent in sorted_sentences:
            sent_tokens = len(sent.split())
            if total_tokens + sent_tokens <= max_tokens:
                selected.append(sent)
                total_tokens += sent_tokens
            else:
                break
                
        return ' '.join(selected)

    def _force_truncate(self, text: str, max_tokens: int) -> str:
        """Fallback truncation with middle preservation"""
        words = text.split()
        if len(words) <= max_tokens:
            return text
            
        keep_start = int(max_tokens * 0.4)
        keep_end = max_tokens - keep_start
        return ' '.join(words[:keep_start] + ['...'] + words[-keep_end:])