"""
RAGPart: Prompt-based verification defense.

Uses the LLM to verify relevance and trustworthiness of retrieved passages
before using them for generation.
"""

import logging
from typing import List, Optional

from .base import BaseDefense

logger = logging.getLogger(__name__)


class RAGPartDefense(BaseDefense):
    """
    RAGPart: Prompt-based passage verification defense.

    Asks the LLM to evaluate each passage for relevance and trustworthiness
    before including it in the context for generation.
    """

    def __init__(
        self,
        generator_model: Optional[object] = None,
        verification_prompt: str = "Is this passage relevant and trustworthy?",
    ):
        """
        Initialize RAGPart defense.

        Args:
            generator_model: LLM generator (OllamaGenerator instance)
            verification_prompt: Prompt for verification
        """
        super().__init__("RAGPart")

        self.generator_model = generator_model
        self.verification_prompt = verification_prompt

        self.num_filtered = 0
        self.total_processed = 0

    def filter_passages(
        self,
        query: str,
        passages: List[str],
        passage_ids: List[str],
    ) -> tuple[List[str], List[str], List[int]]:
        """
        Filter passages based on LLM verification.

        Args:
            query: User query
            passages: Retrieved passages
            passage_ids: Passage IDs

        Returns:
            Filtered passages, IDs, and flagged indices
        """
        if not passages:
            return passages, passage_ids, []

        # If no generator provided, skip filtering
        if self.generator_model is None:
            logger.warning("No generator model provided for RAGPart, skipping defense")
            return passages, passage_ids, []

        filtered_passages = []
        filtered_ids = []
        flagged_indices = []

        for i, (passage, pid) in enumerate(zip(passages, passage_ids)):
            # Verify passage
            is_trustworthy = self._verify_passage(query, passage)

            if is_trustworthy:
                filtered_passages.append(passage)
                filtered_ids.append(pid)
            else:
                flagged_indices.append(i)
                logger.debug(f"Filtered passage {i} (failed verification)")

        self.num_filtered += len(flagged_indices)
        self.total_processed += len(passages)

        return filtered_passages, filtered_ids, flagged_indices

    def _verify_passage(self, query: str, passage: str) -> bool:
        """
        Verify if a passage is relevant and trustworthy.

        Args:
            query: User query
            passage: Passage to verify

        Returns:
            True if passage passes verification
        """
        # Create verification prompt
        verification_query = f"""Given the question: "{query}"

Passage: {passage}

{self.verification_prompt}

Answer with ONLY 'yes' or 'no'."""

        try:
            # Use generator to verify
            import ollama

            response = ollama.generate(
                model="mistral:7b-instruct", prompt=verification_query
            )

            answer = response["response"].strip().lower()

            # Check if answer contains 'yes'
            return "yes" in answer

        except Exception as e:
            logger.warning(f"Verification error: {e}")
            return True  # Default to keeping passage on error

    def get_stats(self) -> dict:
        """Return defense statistics."""
        return {
            "name": self.name,
            "num_filtered": self.num_filtered,
            "total_processed": self.total_processed,
            "filter_rate": (
                self.num_filtered / self.total_processed
                if self.total_processed > 0
                else 0.0
            ),
        }
