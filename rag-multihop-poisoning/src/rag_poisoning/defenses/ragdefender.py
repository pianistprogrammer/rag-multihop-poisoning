"""
RAGDefender: Semantic similarity anomaly detection defense.

Detects poisoned passages by identifying semantic outliers
in the retrieved set.
"""

import logging
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..utils import get_device
from .base import BaseDefense

logger = logging.getLogger(__name__)


class RAGDefenderDefense(BaseDefense):
    """
    RAGDefender: Semantic anomaly detection defense.

    Filters passages that have abnormally high similarity to the query
    compared to the corpus distribution (potential targeted poisoning).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_model: str = "facebook/contriever-msmarco",
        device: str = None,
    ):
        """
        Initialize RAGDefender.

        Args:
            similarity_threshold: Max allowed query-passage similarity
            embedding_model: Model for computing embeddings
            device: Device for computation
        """
        super().__init__("RAGDefender")

        self.similarity_threshold = similarity_threshold

        # Setup device
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(
            embedding_model, device=str(self.device)
        )

        self.num_filtered = 0
        self.total_processed = 0

    def filter_passages(
        self,
        query: str,
        passages: List[str],
        passage_ids: List[str],
    ) -> tuple[List[str], List[str], List[int]]:
        """
        Filter passages based on semantic similarity.

        Args:
            query: User query
            passages: Retrieved passages
            passage_ids: Passage IDs

        Returns:
            Filtered passages, IDs, and flagged indices
        """
        if not passages:
            return passages, passage_ids, []

        # Encode query and passages
        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]

        passage_embeddings = self.embedding_model.encode(
            passages, convert_to_numpy=True, normalize_embeddings=True
        )

        # Compute similarities
        similarities = np.dot(passage_embeddings, query_embedding)

        # Filter based on threshold
        filtered_passages = []
        filtered_ids = []
        flagged_indices = []

        for i, (passage, pid, sim) in enumerate(
            zip(passages, passage_ids, similarities)
        ):
            if sim <= self.similarity_threshold:
                filtered_passages.append(passage)
                filtered_ids.append(pid)
            else:
                flagged_indices.append(i)
                logger.debug(
                    f"Filtered passage {i} (similarity: {sim:.4f} > {self.similarity_threshold})"
                )

        self.num_filtered += len(flagged_indices)
        self.total_processed += len(passages)

        return filtered_passages, filtered_ids, flagged_indices

    def get_stats(self) -> dict:
        """Return defense statistics."""
        return {
            "name": self.name,
            "threshold": self.similarity_threshold,
            "num_filtered": self.num_filtered,
            "total_processed": self.total_processed,
            "filter_rate": (
                self.num_filtered / self.total_processed
                if self.total_processed > 0
                else 0.0
            ),
        }
