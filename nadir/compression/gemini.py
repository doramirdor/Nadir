import os
import hashlib
from functools import lru_cache
from typing import Optional

import google.generativeai as genai
from transformers import pipeline

from nadir.compression import BaseCompression


class GeminiCompressor(BaseCompression):
    """
    A flexible compressor that offers multiple methods to reduce prompt length:
    - Truncation
    - Keyword extraction via a local Hugging Face pipeline
    - High-quality compression via Gemini (Google Generative AI)
    - Automatic fallback logic
    """

    def __init__(
        self,
        gemini_model: Optional[str] = "gemini-1.5-flash", 
        hf_keyword_model: str = "dslim/bert-base-NER",
        cache_size: int = 1000,
        google_api_key: Optional[str] = None
    ):
        """
        :param gemini_model: Name of the Gemini model for high-quality compression (set to None to skip LLM usage).
        :param hf_keyword_model: Hugging Face model identifier for keyword extraction pipeline.
        :param cache_size: Max entries to store in the LRU cache for repeated prompt compression calls.
        :param google_api_key: Your Google Generative AI API key (falls back to env var if None).
        """
        self.gemini_model = gemini_model
        self.hf_keyword_model = hf_keyword_model
        self.cache_size = cache_size

        # Configure Google Generative AI (Gemini)
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if self.gemini_model and not self.google_api_key:
            raise ValueError("A Google Generative AI API key is required for Gemini compression.")
        if self.google_api_key:
            genai.configure(api_key=self.google_api_key)
            self.gemini_model_instance = genai.GenerativeModel(self.gemini_model)

        # Local keyword extraction pipeline
        self.keyword_extractor = pipeline("token-classification", model=self.hf_keyword_model)

    @lru_cache(maxsize=None)
    def compress(
        self,
        prompt: str,
        method: str = "auto",
        max_tokens: int = 200
    ) -> str:
        """
        Compress prompt using the specified method:
          - 'auto': Chooses automatically (truncate -> keywords -> LLM).
          - 'truncate': Simple half-and-half truncation.
          - 'keywords': Keyword extraction.
          - 'high_quality': Gemini-based compression.
        
        :param prompt: The text to compress
        :param method: Which method to use (default = 'auto')
        :param max_tokens: Target maximum words/tokens for final output
        :return: Compressed text
        """
        # If it's already short, no need to compress
        if len(prompt.split()) <= max_tokens:
            return prompt

        if method == "auto":
            return self._auto_compress_prompt(prompt, max_tokens)
        elif method == "truncate":
            return self._truncate_prompt(prompt, max_tokens)
        elif method == "keywords":
            return self._extract_prompt_keywords(prompt)
        elif method == "high_quality":
            return self._compress_prompt_with_llm(prompt, max_tokens)
        else:
            raise ValueError(f"Unknown compression method: {method}")

    def _auto_compress_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Automatically choose compression strategy:
        1) Truncate
        2) If still too long, extract keywords
        3) If STILL too long and we have a Gemini model configured, do LLM compression
        """
        # Step 1: Truncate
        compressed = self._truncate_prompt(prompt, max_tokens)
        if len(compressed.split()) > max_tokens:
            compressed = self._extract_prompt_keywords(compressed)
        if len(compressed.split()) > max_tokens and self.gemini_model:
            compressed = self._compress_prompt_with_llm(compressed, max_tokens)
        return compressed

    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Ultra-fast, simple truncation.
        Takes half the words from the start and half from the end.
        """
        words = prompt.split()
        if len(words) <= max_tokens:
            return prompt
        half = max_tokens // 2
        return " ".join(words[:half] + words[-half:])

    def _extract_prompt_keywords(self, prompt: str) -> str:
        """
        Fast keyword extraction using a local Hugging Face NER-like pipeline.
        Only returns tokens with score > 0.5
        """
        entities = self.keyword_extractor(prompt)
        keywords = [e["word"] for e in entities if e["score"] > 0.5]
        if not keywords:
            # fallback if no keywords found
            return self._truncate_prompt(prompt, 50)
        return " ".join(keywords)

    def _compress_prompt_with_llm(self, prompt: str, max_tokens: int) -> str:
        """
        High-quality compression using Gemini (Google Generative AI).
        Preserves key information, limiting final text to ~max_tokens words.
        """
        if not self.gemini_model:
            raise ValueError("No Gemini model specified for LLM-based compression.")

        try:
            # Build a short system-like instruction
            system_instructions = (
                f"Compress the following text to fewer than {max_tokens} words, "
                "retaining only critical info:\n\n"
            )
            full_prompt = system_instructions + prompt

            # Call Gemini
            response = self.gemini_model_instance.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 1024,
                }
            )
            return response.text.strip()

        except Exception as e:
            print(f"Gemini compression failed: {e}")
            # Fallback to truncation if LLM fails
            return self._truncate_prompt(prompt, max_tokens)
