"""
Test script - runs a single experimental condition for validation.

Usage:
    uv run python experiments/test_single.py --dataset nq --attack poisonedrag --defense none
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_poisoning.attacks.poisonedrag import PoisonedRAGAttack
from rag_poisoning.attacks.coe import ChainOfEvidenceAttack
from rag_poisoning.data.datasets import BEIRDataset
from rag_poisoning.defenses.base import NoDefense
from rag_poisoning.models.retriever import DenseRetriever
from rag_poisoning.models.generator import OllamaGenerator
from rag_poisoning.pipeline.rag_pipeline import RAGPipeline
from rag_poisoning.utils import seed_everything, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Test single RAG poisoning condition"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nq",
        help="Dataset name",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="poisonedrag",
        choices=["poisonedrag", "coe"],
        help="Attack to test",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of test queries",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Volumes/LLModels/Datasets/RAG",
        help="Data directory",
    )

    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    seed_everything(42)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    data_path = Path(args.data_dir) / f"{args.dataset}_processed"
    dataset = BEIRDataset.load(str(data_path))
    logger.info(f"Loaded: {len(dataset.corpus)} docs, {len(dataset.queries)} queries")

    # Sample queries
    import random

    query_ids = random.sample(
        list(dataset.queries.keys()), min(args.num_queries, len(dataset.queries))
    )

    # Initialize attack
    logger.info(f"Initializing attack: {args.attack}")
    if args.attack == "poisonedrag":
        attack = PoisonedRAGAttack()
    elif args.attack == "coe":
        attack = ChainOfEvidenceAttack()

    # Test attack generation
    logger.info("\nTesting attack generation...")
    for query_id in query_ids:
        query = dataset.queries[query_id]
        target = "Test answer"

        logger.info(f"\nQuery: {query}")

        poisoned = attack.generate_poisoned_passages(
            query, target, dataset.corpus, num_passages=3
        )

        logger.info(f"Generated {len(poisoned)} poisoned passages")
        for i, (pid, ptext) in enumerate(poisoned.items(), 1):
            logger.info(f"\nPoison {i}:")
            logger.info(f"  ID: {pid}")
            logger.info(f"  Text: {ptext[:200]}...")

    # Initialize RAG pipeline
    logger.info("\n" + "=" * 80)
    logger.info("Testing RAG Pipeline")
    logger.info("=" * 80)

    retriever = DenseRetriever(model_name="facebook/contriever-msmarco")
    retriever.build_index(dataset.corpus, chunk_size=100)

    generator = OllamaGenerator(model_name="mistral:7b-instruct")

    pipeline = RAGPipeline(retriever, generator, dataset.corpus)

    # Test query
    test_query = dataset.queries[query_ids[0]]
    logger.info(f"\nTest query: {test_query}")

    answer = pipeline.query(test_query, top_k=5)
    logger.info(f"Answer: {answer}")

    logger.info("\n" + "=" * 80)
    logger.info("Test Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
