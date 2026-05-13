"""
Script to download and prepare all datasets for RAG poisoning experiments.

Usage:
    uv run python experiments/prepare_data.py --datasets nq hotpotqa --num-queries 200
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_poisoning.data.datasets import get_dataset_loader
from src.rag_poisoning.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for RAG poisoning"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["nq", "hotpotqa", "2wikimultihop", "musique", "all"],
        default=["all"],
        help="Datasets to prepare",
    )
    parser.add_argument(
        "--num-queries", type=int, default=200, help="Number of queries per dataset"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="C:\\Users\\jerem\\Documents\\Datasets\\RAG",
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Expand "all" to all datasets
    if "all" in args.datasets:
        datasets = ["nq", "hotpotqa", "2wikimultihop", "musique"]
    else:
        datasets = args.datasets

    logger.info(f"Preparing datasets: {datasets}")
    logger.info(f"Number of queries per dataset: {args.num_queries}")
    logger.info(f"Output directory: {args.output_dir}")

    # Prepare each dataset
    for dataset_name in datasets:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing {dataset_name.upper()}")
        logger.info(f"{'=' * 80}")

        try:
            loader_fn = get_dataset_loader(dataset_name)
            dataset = loader_fn(
                output_dir=args.output_dir,
                num_queries=args.num_queries,
                seed=args.seed,
            )

            logger.info(
                f"✓ {dataset_name}: {len(dataset.corpus)} docs, "
                f"{len(dataset.queries)} queries, {len(dataset.qrels)} qrels"
            )

        except Exception as e:
            logger.error(f"✗ Failed to process {dataset_name}: {e}", exc_info=True)

    logger.info("\n" + "=" * 80)
    logger.info("Dataset preparation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
