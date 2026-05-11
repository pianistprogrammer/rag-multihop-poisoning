"""
Main experiment entry point - runs the full experimental grid.

Usage:
    uv run python experiments/run_grid.py --datasets nq hotpotqa --num-queries 200
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_poisoning.data.datasets import BEIRDataset
from rag_poisoning.evaluation.runner import ExperimentRunner
from rag_poisoning.models.retriever import DenseRetriever
from rag_poisoning.models.generator import OllamaGenerator
from rag_poisoning.pipeline.rag_pipeline import RAGPipeline
from rag_poisoning.utils import seed_everything, setup_logging, get_model_device_info


def main():
    parser = argparse.ArgumentParser(
        description="Run full RAG poisoning experimental grid"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["nq", "hotpotqa", "2wikimultihop", "musique", "all"],
        default=["nq"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=["poisonedrag", "corruptrag", "pidp", "coe", "all"],
        default=["all"],
        help="Attacks to evaluate",
    )
    parser.add_argument(
        "--defenses",
        nargs="+",
        choices=["none", "filterrag", "ragdefender", "ragpart", "ragmask", "all"],
        default=["all"],
        help="Defenses to evaluate",
    )
    parser.add_argument(
        "--num-poisoned",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Number of poisoned passages per query",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=200,
        help="Number of queries per dataset",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Volumes/LLModels/Datasets/RAG",
        help="Directory containing processed datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--retriever-model",
        type=str,
        default="facebook/contriever-msmarco",
        help="Retriever model name",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="mistral:7b-instruct",
        help="Ollama generator model name",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-trackio", action="store_true", help="Disable Trackio logging"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available (automatically enabled)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    seed_everything(args.seed)

    # Log device info
    device_info = get_model_device_info()
    logger.info("Device Information:")
    for key, value in device_info.items():
        logger.info(f"  {key}: {value}")

    # Expand "all" options
    if "all" in args.datasets:
        dataset_names = ["nq", "hotpotqa", "2wikimultihop", "musique"]
    else:
        dataset_names = args.datasets

    if "all" in args.attacks:
        attack_names = ["poisonedrag", "corruptrag", "pidp", "coe"]
    else:
        attack_names = args.attacks

    if "all" in args.defenses:
        defense_names = ["none", "filterrag", "ragdefender", "ragpart", "ragmask"]
    else:
        defense_names = args.defenses

    # Load datasets
    logger.info("\n" + "=" * 80)
    logger.info("Loading Datasets")
    logger.info("=" * 80)

    datasets = {}
    for dataset_name in dataset_names:
        try:
            data_path = Path(args.data_dir) / f"{dataset_name}_processed"
            logger.info(f"Loading {dataset_name} from {data_path}...")
            dataset = BEIRDataset.load(str(data_path))
            datasets[dataset_name] = dataset
            logger.info(
                f"  ✓ {dataset_name}: {len(dataset.corpus)} docs, "
                f"{len(dataset.queries)} queries"
            )
        except Exception as e:
            logger.error(f"  ✗ Failed to load {dataset_name}: {e}")
            sys.exit(1)

    # Initialize retriever (use first dataset for initial index)
    logger.info("\n" + "=" * 80)
    logger.info("Initializing RAG Pipeline")
    logger.info("=" * 80)

    first_dataset = datasets[dataset_names[0]]

    logger.info(f"Initializing retriever: {args.retriever_model}")
    retriever = DenseRetriever(model_name=args.retriever_model)
    retriever.build_index(first_dataset.corpus, chunk_size=100)

    logger.info(f"Initializing generator: {args.generator_model}")
    generator = OllamaGenerator(model_name=args.generator_model)

    # Create pipeline
    pipeline = RAGPipeline(retriever, generator, first_dataset.corpus)

    # Initialize experiment runner
    logger.info("\n" + "=" * 80)
    logger.info("Initializing Experiment Runner")
    logger.info("=" * 80)

    runner = ExperimentRunner(
        datasets=datasets,
        retriever=retriever,
        pipeline=pipeline,
        output_dir=args.output_dir,
        use_trackio=not args.no_trackio,
    )

    # Compute total conditions
    total_conditions = (
        len(dataset_names)
        * len(attack_names)
        * len(defense_names)
        * len(args.num_poisoned)
    )

    logger.info(f"Experiment Grid Configuration:")
    logger.info(f"  Datasets: {dataset_names}")
    logger.info(f"  Attacks: {attack_names}")
    logger.info(f"  Defenses: {defense_names}")
    logger.info(f"  Poisoned per query: {args.num_poisoned}")
    logger.info(f"  Queries per dataset: {args.num_queries}")
    logger.info(f"  Total conditions: {total_conditions}")
    logger.info(f"  Total evaluations: {total_conditions * args.num_queries}")

    # Run grid
    logger.info("\n" + "=" * 80)
    logger.info("Starting Experimental Grid")
    logger.info("=" * 80)

    runner.run_grid(
        dataset_names=dataset_names,
        attack_names=attack_names,
        defense_names=defense_names,
        num_poisoned_list=args.num_poisoned,
        num_queries=args.num_queries,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Experiment Complete!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {args.output_dir}/all_results.json")


if __name__ == "__main__":
    main()
