"""
End-to-end RAG pipeline combining retrieval and generation.
"""

import logging
from typing import Dict, List, Optional, Tuple

from ..data.datasets import BEIRDataset
from ..models.generator import OllamaGenerator
from ..models.retriever import DenseRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline: query → retrieve → generate → answer.
    """

    def __init__(
        self,
        retriever: DenseRetriever,
        generator: OllamaGenerator,
        corpus: Dict[str, Dict[str, str]],
    ):
        """
        Initialize RAG pipeline.

        Args:
            retriever: Dense retriever instance
            generator: Generator instance
            corpus: Document corpus {doc_id: {"title": str, "text": str}}
        """
        self.retriever = retriever
        self.generator = generator
        self.corpus = corpus

    def query(
        self,
        question: str,
        top_k: int = 5,
        return_contexts: bool = False,
        return_doc_ids: bool = False,
    ) -> str | Tuple[str, List[str]] | Tuple[str, List[str], List[str]]:
        """
        Run end-to-end RAG query.

        Args:
            question: User question
            top_k: Number of documents to retrieve
            return_contexts: Whether to return retrieved contexts
            return_doc_ids: Whether to return retrieved doc IDs

        Returns:
            Generated answer, optionally with contexts and doc IDs
        """
        # Retrieve
        doc_ids, scores = self.retriever.retrieve(
            question, top_k=top_k, return_scores=True
        )

        # Get full text for retrieved documents
        contexts = []
        for doc_id in doc_ids:
            # Handle chunked doc IDs
            base_doc_id = doc_id.split("_chunk_")[0]
            if base_doc_id in self.corpus:
                doc = self.corpus[base_doc_id]
                title = doc.get("title", "")
                text = doc.get("text", "")
                full_text = f"{title}. {text}".strip() if title else text
                contexts.append(full_text)
            else:
                logger.warning(f"Document {doc_id} not found in corpus")
                contexts.append("[Document not found]")

        # Generate
        answer = self.generator.generate(question, contexts)

        # Return based on flags
        if return_contexts and return_doc_ids:
            return answer, contexts, doc_ids
        elif return_contexts:
            return answer, contexts
        elif return_doc_ids:
            return answer, doc_ids
        return answer

    def batch_query(
        self,
        questions: List[str],
        top_k: int = 5,
    ) -> List[str]:
        """
        Run batch RAG queries.

        Args:
            questions: List of questions
            top_k: Number of documents per query

        Returns:
            List of generated answers
        """
        answers = []
        for question in questions:
            answer = self.query(question, top_k=top_k)
            answers.append(answer)
        return answers


def build_rag_pipeline(
    dataset: BEIRDataset,
    retriever_model: str = "facebook/contriever-msmarco",
    generator_model: str = "mistral:7b-instruct",
    chunk_size: int = 100,
    device: Optional[str] = None,
) -> RAGPipeline:
    """
    Build a complete RAG pipeline from dataset.

    Args:
        dataset: BEIRDataset instance
        retriever_model: Hugging Face retriever model name
        generator_model: Ollama generator model name
        chunk_size: Chunk size for document splitting
        device: Device for retriever ('cpu', 'cuda', 'mps')

    Returns:
        Initialized RAGPipeline
    """
    logger.info("Building RAG pipeline...")

    # Initialize retriever
    logger.info(f"Initializing retriever: {retriever_model}")
    retriever = DenseRetriever(model_name=retriever_model, device=device)

    # Build index
    logger.info("Building retrieval index...")
    retriever.build_index(dataset.corpus, chunk_size=chunk_size)

    # Initialize generator
    logger.info(f"Initializing generator: {generator_model}")
    generator = OllamaGenerator(model_name=generator_model)

    # Create pipeline
    pipeline = RAGPipeline(retriever, generator, dataset.corpus)

    logger.info("RAG pipeline ready!")
    return pipeline
