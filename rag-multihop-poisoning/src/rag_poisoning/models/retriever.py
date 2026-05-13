"""
Dense retrieval model wrapper using sentence-transformers and FAISS.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..utils import get_device

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense passage retriever using sentence-transformers and FAISS.

    Supports models like Contriever, BGE-base, etc.
    """

    def __init__(
        self,
        model_name: str = "facebook/contriever-msmarco",
        index_type: str = "flat",
        device: Optional[str] = None,
    ):
        """
        Initialize retriever.

        Args:
            model_name: Hugging Face model name
            index_type: FAISS index type ('flat', 'hnsw')
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.model_name = model_name
        self.index_type = index_type

        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        logger.info(f"Loading retriever model: {model_name}")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.dimension = self.model.get_embedding_dimension()

        self.index: Optional[faiss.Index] = None
        self.doc_ids: List[str] = []

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: L2 normalize embeddings

        Returns:
            Numpy array of shape (len(texts), dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings

    def build_index(
        self,
        corpus: Dict[str, Dict[str, str]],
        batch_size: int = 32,
        chunk_size: int = 100,
        cache_dir: Optional[str] = None,
        save_every: int = 10000,
    ) -> None:
        """
        Build FAISS index from corpus with incremental caching.

        Args:
            corpus: Dict[doc_id, {"title": str, "text": str}]
            batch_size: Batch size for encoding
            chunk_size: Number of words per chunk (for chunking long docs)
            cache_dir: Directory to save partial index progress (enables incremental caching)
            save_every: Save progress every N passages encoded
        """
        logger.info(f"Building index for {len(corpus)} documents...")

        # Check if we can resume from cache
        if cache_dir:
            # Scope cache to this model so different retrievers never share files
            model_slug = self.model_name.replace("/", "_").replace(":", "_")
            cache_path = Path(cache_dir) / model_slug
            cache_path.mkdir(parents=True, exist_ok=True)
            partial_embeddings_file = cache_path / "partial_embeddings.npy"
            partial_doc_ids_file = cache_path / "partial_doc_ids.npy"
            partial_texts_file = cache_path / "partial_texts.json"

            if partial_embeddings_file.exists() and partial_doc_ids_file.exists():
                logger.info("Found partial index cache - resuming...")
                try:
                    # Load partial progress
                    existing_embeddings = np.load(partial_embeddings_file)
                    existing_doc_ids = np.load(partial_doc_ids_file, allow_pickle=True).tolist()

                    with open(partial_texts_file) as f:
                        import json
                        existing_texts = json.load(f)

                    num_cached = len(existing_doc_ids)
                    logger.info(f"Loaded {num_cached} cached passages - will continue from there")

                    # We'll continue below after preparing all docs
                    resume_from = num_cached
                except Exception as e:
                    logger.warning(f"Failed to load partial cache: {e}")
                    resume_from = 0
                    existing_embeddings = None
                    existing_doc_ids = []
                    existing_texts = []
            else:
                resume_from = 0
                existing_embeddings = None
                existing_doc_ids = []
                existing_texts = []
        else:
            resume_from = 0
            existing_embeddings = None
            existing_doc_ids = []
            existing_texts = []

        # Prepare documents (title + text)
        doc_texts = []
        self.doc_ids = []

        for doc_id, doc in tqdm(corpus.items(), desc="Preparing docs"):
            title = doc.get("title", "")
            text = doc.get("text", "")

            # Combine title and text
            full_text = f"{title}. {text}".strip()

            # Optional: chunk long documents
            words = full_text.split()
            if len(words) > chunk_size:
                # Split into chunks
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i : i + chunk_size])
                    doc_texts.append(chunk)
                    self.doc_ids.append(f"{doc_id}_chunk_{i // chunk_size}")
            else:
                doc_texts.append(full_text)
                self.doc_ids.append(doc_id)

        logger.info(f"Total passages after chunking: {len(doc_texts)}")

        # If resuming, check if we already have all passages
        if resume_from > 0:
            if resume_from >= len(doc_texts):
                logger.info("All passages already encoded - using cached embeddings")
                embeddings = existing_embeddings
                self.doc_ids = existing_doc_ids
            else:
                logger.info(f"Encoding remaining passages ({resume_from}/{len(doc_texts)} done)...")
                remaining_texts = doc_texts[resume_from:]

                # Encode in batches with incremental saving
                remaining_embeddings = self._encode_with_incremental_save(
                    remaining_texts,
                    batch_size=batch_size,
                    cache_dir=str(cache_path) if cache_dir else None,
                    save_every=save_every,
                    offset=resume_from,
                    existing_embeddings=existing_embeddings,
                    existing_doc_ids=existing_doc_ids,
                    all_doc_ids=self.doc_ids,
                )

                # Combine with existing
                embeddings = np.vstack([existing_embeddings, remaining_embeddings])
        else:
            # Encode corpus with incremental saving
            logger.info("Encoding corpus...")
            embeddings = self._encode_with_incremental_save(
                doc_texts,
                batch_size=batch_size,
                cache_dir=str(cache_path) if cache_dir else None,
                save_every=save_every,
                offset=0,
                existing_embeddings=None,
                existing_doc_ids=[],
                all_doc_ids=self.doc_ids,
            )

        # Build FAISS index
        logger.info(f"Building {self.index_type} FAISS index...")
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        logger.info(f"Index built with {self.index.ntotal} vectors")


    def _encode_with_incremental_save(
        self,
        texts: List[str],
        batch_size: int,
        cache_dir: Optional[str],
        save_every: int,
        offset: int,
        existing_embeddings: Optional[np.ndarray],
        existing_doc_ids: List[str],
        all_doc_ids: List[str],
    ) -> np.ndarray:
        """
        Encode texts with incremental saving to disk.

        Returns:
            Embeddings for the new texts only
        """
        all_embeddings = []

        # Process in chunks to enable incremental saving
        num_texts = len(texts)

        for start_idx in tqdm(range(0, num_texts, batch_size), desc="Batches"):
            end_idx = min(start_idx + batch_size, num_texts)
            batch_texts = texts[start_idx:end_idx]

            # Encode batch
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            all_embeddings.append(batch_embeddings)

            # Save incrementally every N passages
            if cache_dir and (end_idx % save_every == 0 or end_idx == num_texts):
                current_new_embeddings = np.vstack(all_embeddings)

                # Combine with existing if resuming
                if existing_embeddings is not None:
                    current_total_embeddings = np.vstack([existing_embeddings, current_new_embeddings])
                    current_total_doc_ids = existing_doc_ids + all_doc_ids[offset:offset + end_idx]
                else:
                    current_total_embeddings = current_new_embeddings
                    current_total_doc_ids = all_doc_ids[:end_idx]

                # Save to cache
                cache_path = Path(cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)

                np.save(cache_path / "partial_embeddings.npy", current_total_embeddings)
                np.save(cache_path / "partial_doc_ids.npy", np.array(current_total_doc_ids))

                # Save progress marker
                import json
                with open(cache_path / "partial_texts.json", "w") as f:
                    json.dump({"num_encoded": len(current_total_doc_ids)}, f)

                logger.info(
                    f"✓ Saved progress: {len(current_total_doc_ids)}/{offset + num_texts} passages encoded"
                )

        return np.vstack(all_embeddings)

    def search(
        self, queries: List[str], top_k: int = 5, batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k documents for each query.

        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve per query
            batch_size: Batch size for encoding queries

        Returns:
            scores: Array of shape (len(queries), top_k)
            doc_indices: Array of shape (len(queries), top_k)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Encode queries
        query_embeddings = self.encode(
            queries, batch_size=batch_size, show_progress=False
        )

        # Search
        scores, doc_indices = self.index.search(
            query_embeddings.astype(np.float32), top_k
        )

        return scores, doc_indices

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_scores: bool = False,
    ) -> List[str] | Tuple[List[str], List[float]]:
        """
        Retrieve top-k document IDs for a single query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve
            return_scores: Whether to return scores

        Returns:
            List of document IDs, optionally with scores
        """
        scores, indices = self.search([query], top_k=top_k)

        # Convert indices to doc_ids
        retrieved_doc_ids = [self.doc_ids[idx] for idx in indices[0]]

        if return_scores:
            return retrieved_doc_ids, scores[0].tolist()
        return retrieved_doc_ids

    def save_index(self, save_path: str) -> None:
        """Save FAISS index and doc IDs to disk."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(self.index, str(save_path / "index.faiss"))

        # Save doc IDs
        np.save(save_path / "doc_ids.npy", np.array(self.doc_ids))

        logger.info(f"Index saved to {save_path}")

    def load_index(self, load_path: str) -> None:
        """Load FAISS index and doc IDs from disk."""
        load_path = Path(load_path)

        # Load index
        self.index = faiss.read_index(str(load_path / "index.faiss"))

        # Load doc IDs
        self.doc_ids = np.load(load_path / "doc_ids.npy", allow_pickle=True).tolist()

        logger.info(f"Index loaded from {load_path}")


def inject_poisoned_passages(
    retriever: DenseRetriever,
    poisoned_passages: Dict[str, str],
) -> None:
    """
    Inject poisoned passages into an existing index.

    Args:
        retriever: DenseRetriever instance with built index
        poisoned_passages: Dict[passage_id, passage_text]
    """
    if retriever.index is None:
        raise RuntimeError("Index not built. Call build_index() first.")

    logger.info(f"Injecting {len(poisoned_passages)} poisoned passages...")

    # Encode poisoned passages
    passage_texts = list(poisoned_passages.values())
    passage_ids = list(poisoned_passages.keys())

    embeddings = retriever.encode(passage_texts, show_progress=False)

    # Add to index
    retriever.index.add(embeddings.astype(np.float32))
    retriever.doc_ids.extend(passage_ids)

    logger.info(f"Index now contains {retriever.index.ntotal} vectors")
