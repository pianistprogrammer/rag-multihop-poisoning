"""
RAGMask: Token masking defense.

Masks potentially adversarial tokens in retrieved passages
before feeding them to the generator.
"""

import logging
import random
import re
from typing import List

from .base import BaseDefense

logger = logging.getLogger(__name__)


class RAGMaskDefense(BaseDefense):
    """
    RAGMask: Token masking defense.

    Masks suspicious tokens/patterns in retrieved passages:
    - Prompt injection patterns
    - Directive keywords
    - Excessive repetition
    - Authority claims
    """

    def __init__(self, mask_ratio: float = 0.3):
        """
        Initialize RAGMask defense.

        Args:
            mask_ratio: Ratio of suspicious tokens to mask
        """
        super().__init__("RAGMask")

        self.mask_ratio = mask_ratio

        # Define suspicious patterns
        self.suspicious_patterns = [
            # Directive keywords
            r"\b(must|should|always|never|only|definitively?)\b",
            r"\b(ignore|disregard|supersede|override)\b",
            r"\b(instruction|directive|command|important)\b",
            # Authority claims
            r"\b(according to|verified by|confirmed by|official)\b",
            r"\b(expert|authorit(?:y|ative)|definitive|conclusive)\b",
            # Meta-instructions
            r"\b(system|note to|context:|important:)\b",
            r"\b(respond|answer|output|generate)\b",
            # Repetitive emphasis
            r"\b(correct|accurate|true|fact|certain)\b",
        ]

        self.num_masked = 0
        self.total_processed = 0

    def filter_passages(
        self,
        query: str,
        passages: List[str],
        passage_ids: List[str],
    ) -> tuple[List[str], List[str], List[int]]:
        """
        Mask suspicious content in passages.

        Unlike other defenses, this returns all passages but with
        potentially adversarial content masked.

        Args:
            query: User query
            passages: Retrieved passages
            passage_ids: Passage IDs

        Returns:
            Masked passages, IDs, and flagged indices (passages with masking)
        """
        if not passages:
            return passages, passage_ids, []

        masked_passages = []
        flagged_indices = []

        for i, passage in enumerate(passages):
            masked_passage, was_masked = self._mask_passage(passage)
            masked_passages.append(masked_passage)

            if was_masked:
                flagged_indices.append(i)
                self.num_masked += 1

        self.total_processed += len(passages)

        return masked_passages, passage_ids, flagged_indices

    def _mask_passage(self, passage: str) -> tuple[str, bool]:
        """
        Mask suspicious tokens in a passage.

        Args:
            passage: Input passage

        Returns:
            Tuple of (masked_passage, was_masked)
        """
        original = passage
        was_masked = False

        # Apply pattern-based masking
        for pattern in self.suspicious_patterns:
            matches = list(re.finditer(pattern, passage, re.IGNORECASE))

            if matches:
                # Mask a fraction of matches
                num_to_mask = max(1, int(len(matches) * self.mask_ratio))
                matches_to_mask = random.sample(matches, num_to_mask)

                for match in matches_to_mask:
                    # Replace with [MASKED]
                    start, end = match.span()
                    passage = passage[:start] + "[MASKED]" + passage[end:]
                    was_masked = True

        # Additional heuristic: mask if passage is too repetitive
        words = passage.lower().split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            # Check for excessive repetition
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.15:  # Word appears >15% of time
                # Mask some instances
                most_common = max(word_freq, key=word_freq.get)
                passage = re.sub(
                    rf"\b{re.escape(most_common)}\b",
                    "[MASKED]",
                    passage,
                    count=max_freq // 2,
                    flags=re.IGNORECASE,
                )
                was_masked = True

        if was_masked:
            logger.debug(f"Masked passage: {len(original)} → {len(passage)} chars")

        return passage, was_masked

    def get_stats(self) -> dict:
        """Return defense statistics."""
        return {
            "name": self.name,
            "mask_ratio": self.mask_ratio,
            "num_masked": self.num_masked,
            "total_processed": self.total_processed,
            "mask_rate": (
                self.num_masked / self.total_processed
                if self.total_processed > 0
                else 0.0
            ),
        }
