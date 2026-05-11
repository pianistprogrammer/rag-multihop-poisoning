"""
Base class for RAG defense mechanisms.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List

logger = logging.getLogger(__name__)


class BaseDefense(ABC):
    """
    Abstract base class for RAG defenses.
    """

    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized defense: {name}")

    @abstractmethod
    def filter_passages(
        self,
        query: str,
        passages: List[str],
        passage_ids: List[str],
    ) -> tuple[List[str], List[str], List[int]]:
        """
        Filter retrieved passages to remove potential poisoning.

        Args:
            query: The user query
            passages: List of retrieved passage texts
            passage_ids: List of passage IDs

        Returns:
            Tuple of (filtered_passages, filtered_ids, flagged_indices)
        """
        pass

    def get_stats(self) -> Dict:
        """Return defense statistics."""
        return {"name": self.name}


class NoDefense(BaseDefense):
    """
    Baseline: no defense applied.
    """

    def __init__(self):
        super().__init__("NoDefense")

    def filter_passages(
        self,
        query: str,
        passages: List[str],
        passage_ids: List[str],
    ) -> tuple[List[str], List[str], List[int]]:
        """Return all passages unchanged."""
        return passages, passage_ids, []
