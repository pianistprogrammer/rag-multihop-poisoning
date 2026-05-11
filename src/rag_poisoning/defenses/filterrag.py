"""
FilterRAG: Perplexity-based defense.

Filters out passages with abnormally high perplexity scores,
which may indicate synthetic or adversarial content.
"""

import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils import get_device
from .base import BaseDefense

logger = logging.getLogger(__name__)


class FilterRAGDefense(BaseDefense):
    """
    FilterRAG: Perplexity-based filtering defense.

    Computes perplexity of each retrieved passage and filters
    out those exceeding a threshold (likely synthetic).
    """

    def __init__(
        self,
        perplexity_threshold: float = 50.0,
        model_name: str = "gpt2",
        device: str = None,
    ):
        """
        Initialize FilterRAG defense.

        Args:
            perplexity_threshold: Max allowed perplexity
            model_name: Language model for perplexity computation
            device: Device for computation
        """
        super().__init__("FilterRAG")

        self.perplexity_threshold = perplexity_threshold

        # Setup device
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        # Load model for perplexity computation
        logger.info(f"Loading perplexity model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_filtered = 0
        self.total_processed = 0

    def filter_passages(
        self,
        query: str,
        passages: List[str],
        passage_ids: List[str],
    ) -> tuple[List[str], List[str], List[int]]:
        """
        Filter passages based on perplexity.

        Args:
            query: User query
            passages: Retrieved passages
            passage_ids: Passage IDs

        Returns:
            Filtered passages, IDs, and flagged indices
        """
        if not passages:
            return passages, passage_ids, []

        # Compute perplexity for each passage
        perplexities = []
        for passage in passages:
            ppl = self._compute_perplexity(passage)
            perplexities.append(ppl)

        # Filter based on threshold
        filtered_passages = []
        filtered_ids = []
        flagged_indices = []

        for i, (passage, pid, ppl) in enumerate(
            zip(passages, passage_ids, perplexities)
        ):
            if ppl <= self.perplexity_threshold:
                filtered_passages.append(passage)
                filtered_ids.append(pid)
            else:
                flagged_indices.append(i)
                logger.debug(
                    f"Filtered passage {i} (perplexity: {ppl:.2f} > {self.perplexity_threshold})"
                )

        self.num_filtered += len(flagged_indices)
        self.total_processed += len(passages)

        return filtered_passages, filtered_ids, flagged_indices

    def _compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of text.

        Args:
            text: Input text

        Returns:
            Perplexity score
        """
        try:
            # Tokenize
            encodings = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            input_ids = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)

            # Compute loss
            with torch.no_grad():
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=input_ids
                )
                loss = outputs.loss

            # Perplexity = exp(loss)
            perplexity = torch.exp(loss).item()

            return perplexity

        except Exception as e:
            logger.warning(f"Error computing perplexity: {e}")
            return 0.0  # Return low perplexity on error (don't filter)

    def get_stats(self) -> dict:
        """Return defense statistics."""
        return {
            "name": self.name,
            "threshold": self.perplexity_threshold,
            "num_filtered": self.num_filtered,
            "total_processed": self.total_processed,
            "filter_rate": (
                self.num_filtered / self.total_processed
                if self.total_processed > 0
                else 0.0
            ),
        }
