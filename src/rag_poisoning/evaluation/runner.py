"""
Experiment runner for RAG poisoning evaluation.

Runs the full experimental grid:
4 datasets × 4 attacks × 4 defenses × 3 injection levels = 192 conditions
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import trackio
from tqdm import tqdm

from ..attacks.base import BaseAttack
from ..attacks.coe import ChainOfEvidenceAttack
from ..attacks.corruptrag import CorruptRAGAttack
from ..attacks.pidp import PIDPAttack
from ..attacks.poisonedrag import PoisonedRAGAttack
from ..data.datasets import BEIRDataset
from ..defenses.base import BaseDefense, NoDefense
from ..defenses.filterrag import FilterRAGDefense
from ..defenses.ragdefender import RAGDefenderDefense
from ..defenses.ragmask import RAGMaskDefense
from ..defenses.ragpart import RAGPartDefense
from ..models.retriever import DenseRetriever, inject_poisoned_passages
from ..pipeline.rag_pipeline import RAGPipeline
from .metrics import MetricsTracker, evaluate_single_condition

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    dataset_name: str
    attack_name: str
    defense_name: str
    num_poisoned: int
    num_queries: int = 200
    top_k: int = 5
    seed: int = 42


class ExperimentRunner:
    """
    Orchestrates the full experimental grid.
    """

    def __init__(
        self,
        datasets: Dict[str, BEIRDataset],
        retriever: DenseRetriever,
        pipeline: RAGPipeline,
        output_dir: str = "./results",
        use_trackio: bool = True,
        trackio_project: str = "rag-multihop-poisoning",
    ):
        """
        Initialize experiment runner.

        Args:
            datasets: Dict of dataset_name → BEIRDataset
            retriever: Dense retriever instance
            pipeline: RAG pipeline instance
            output_dir: Directory for saving results
            use_trackio: Whether to use Trackio logging
            trackio_project: Trackio project name
        """
        self.datasets = datasets
        self.retriever = retriever
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_trackio = use_trackio
        if self.use_trackio:
            trackio.init(project=trackio_project)

        self.metrics_tracker = MetricsTracker()

        # Initialize attacks and defenses
        self.attacks = self._init_attacks()
        self.defenses = self._init_defenses()

        # Checkpoint file for resume functionality
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.completed_experiments: Set[str] = self._load_checkpoint()

    def _init_attacks(self) -> Dict[str, BaseAttack]:
        """Initialize all attack methods."""
        return {
            "poisonedrag": PoisonedRAGAttack(),
            "corruptrag": CorruptRAGAttack(),
            "pidp": PIDPAttack(),
            "coe": ChainOfEvidenceAttack(),
        }

    def _init_defenses(self) -> Dict[str, BaseDefense]:
        """Initialize all defense methods."""
        return {
            "none": NoDefense(),
            "filterrag": FilterRAGDefense(),
            "ragdefender": RAGDefenderDefense(),
            "ragpart": RAGPartDefense(),
            "ragmask": RAGMaskDefense(),
        }

    def _load_checkpoint(self) -> Set[str]:
        """
        Load checkpoint of completed experiments.

        Returns:
            Set of completed experiment IDs
        """
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint found - starting fresh")
            return set()

        try:
            with open(self.checkpoint_file, "r") as f:
                checkpoint = json.load(f)
                completed = set(checkpoint.get("completed_experiments", []))
                logger.info(
                    f"Loaded checkpoint: {len(completed)} experiments already completed"
                )

                # Also load previous results into metrics tracker
                if "results" in checkpoint:
                    self.metrics_tracker.results = checkpoint["results"]
                    logger.info(f"Restored {len(checkpoint['results'])} previous results")

                return completed
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return set()

    def _save_checkpoint(self, experiment_id: str, partial: bool = False):
        """
        Save checkpoint after completing an experiment.

        Args:
            experiment_id: ID of completed experiment
            partial: If True, saves as partial (in-progress) checkpoint
        """
        if not partial:
            self.completed_experiments.add(experiment_id)

        checkpoint = {
            "completed_experiments": list(self.completed_experiments),
            "results": self.metrics_tracker.results,
            "last_updated": time.time(),
        }

        # Add partial status if in-progress
        if partial:
            checkpoint["partial_experiment"] = experiment_id

        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)
            logger.debug(f"Checkpoint saved: {experiment_id} (partial={partial})")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _get_poison_cache_path(self, config: ExperimentConfig) -> Path:
        """Get cache file path for poisoned passages."""
        cache_dir = self.output_dir / "poison_cache"
        cache_dir.mkdir(exist_ok=True)

        cache_file = (
            f"{config.dataset_name}_{config.attack_name}_"
            f"{config.num_poisoned}_{config.num_queries}.json"
        )
        return cache_dir / cache_file

    def _get_index_cache_path(self, dataset_name: str) -> Path:
        """Get cache directory for retriever index."""
        cache_dir = self.output_dir / "index_cache" / dataset_name
        return cache_dir

    def _load_poison_cache(self, config: ExperimentConfig) -> Optional[Dict]:
        """Load cached poisoned passages if available."""
        cache_path = self._get_poison_cache_path(config)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                cache = json.load(f)
                num_queries = len(cache.get("target_answers", []))
                logger.info(
                    f"Loaded poisoned passages from cache: {cache_path.name} "
                    f"({num_queries}/{config.num_queries} queries)"
                )
                return cache
        except Exception as e:
            logger.warning(f"Failed to load poison cache: {e}")
            return None

    def _save_poison_cache(self, config: ExperimentConfig, data: Dict, partial: bool = False):
        """Save poisoned passages to cache."""
        cache_path = self._get_poison_cache_path(config)

        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)

            num_queries = len(data.get("target_answers", []))
            status = f"partial {num_queries}/{config.num_queries}" if partial else "complete"
            logger.info(f"Saved poisoned passages to cache: {cache_path.name} ({status})")
        except Exception as e:
            logger.warning(f"Failed to save poison cache: {e}")

    def run_single_experiment(
        self, config: ExperimentConfig
    ) -> Dict[str, float]:
        """
        Run a single experimental condition.

        Args:
            config: Experiment configuration

        Returns:
            Dict of metric results
        """
        logger.info(
            f"Running: {config.dataset_name} | {config.attack_name} | "
            f"{config.defense_name} | {config.num_poisoned} poisoned"
        )

        # Get dataset
        dataset = self.datasets[config.dataset_name]

        # Check if we need to rebuild index for this dataset
        index_cache_path = self._get_index_cache_path(config.dataset_name)
        if index_cache_path.exists():
            logger.info(f"Loading cached index for {config.dataset_name}...")
            try:
                self.retriever.load_index(str(index_cache_path))
                logger.info(f"✓ Loaded cached index ({self.retriever.index.ntotal} vectors)")
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}")
                logger.info(f"Rebuilding index for {config.dataset_name}...")
                self.retriever.build_index(dataset.corpus, chunk_size=100)
                self.retriever.save_index(str(index_cache_path))
                logger.info(f"✓ Index cached to {index_cache_path}")
        else:
            logger.info(f"Building index for {config.dataset_name}...")
            self.retriever.build_index(dataset.corpus, chunk_size=100)
            self.retriever.save_index(str(index_cache_path))
            logger.info(f"✓ Index cached to {index_cache_path}")

        # Sample queries
        import random

        random.seed(config.seed)
        query_ids = random.sample(
            list(dataset.queries.keys()),
            min(config.num_queries, len(dataset.queries)),
        )

        # Get attack and defense
        attack = self.attacks[config.attack_name]
        defense = self.defenses[config.defense_name]

        # Try to load cached poisoned passages
        poison_cache = self._load_poison_cache(config)

        if poison_cache:
            # Use cached data
            poisoned_passages_all = poison_cache["poisoned_passages"]
            poisoned_doc_ids_per_query = poison_cache["poisoned_doc_ids_per_query"]
            target_answers = poison_cache["target_answers"]

            # Check if cache is complete
            cached_queries = len(target_answers)
            if cached_queries >= config.num_queries:
                logger.info(
                    f"Using complete cached poisoned passages ({cached_queries} queries)"
                )
            else:
                logger.info(
                    f"Using partial cached poisoned passages ({cached_queries}/{config.num_queries} queries) - "
                    f"will continue generation from query {cached_queries}"
                )

                # Get already processed query IDs
                processed_query_ids = query_ids[:cached_queries]
                remaining_query_ids = query_ids[cached_queries:]

                # Continue generation for remaining queries
                for idx, query_id in enumerate(
                    tqdm(remaining_query_ids, desc="Continuing poisoned passage generation")
                ):
                    query = dataset.queries[query_id]

                    # Get ground truth answer
                    if query_id in dataset.qrels and dataset.qrels[query_id]:
                        gt_doc_id = list(dataset.qrels[query_id].keys())[0]
                        gt_doc = dataset.corpus.get(gt_doc_id, {})
                        target_answer = gt_doc.get("text", "Unknown")[:100]
                    else:
                        target_answer = "Synthetic target answer"

                    target_answers.append(target_answer)

                    # Generate poisoned passages
                    poisoned_passages = attack.generate_poisoned_passages(
                        query=query,
                        target_answer=target_answer,
                        corpus=dataset.corpus,
                        num_passages=config.num_poisoned,
                    )

                    # Track poisoned doc IDs
                    poisoned_doc_ids_per_query.append(list(poisoned_passages.keys()))

                    # Add to global dict
                    poisoned_passages_all.update(poisoned_passages)

                    # Save incrementally every 10 queries
                    if (idx + 1) % 10 == 0 or (idx + 1) == len(remaining_query_ids):
                        cache_data = {
                            "poisoned_passages": poisoned_passages_all,
                            "poisoned_doc_ids_per_query": poisoned_doc_ids_per_query,
                            "target_answers": target_answers,
                            "config": {
                                "dataset": config.dataset_name,
                                "attack": config.attack_name,
                                "num_poisoned": config.num_poisoned,
                                "num_queries": config.num_queries,
                            },
                        }
                        is_partial = len(target_answers) < config.num_queries
                        self._save_poison_cache(config, cache_data, partial=is_partial)
        else:
            # Generate poisoned passages from scratch
            poisoned_passages_all = {}
            poisoned_doc_ids_per_query = []
            target_answers = []

            for idx, query_id in enumerate(
                tqdm(query_ids, desc="Generating poisoned passages")
            ):
                query = dataset.queries[query_id]

                # Get ground truth answer (from first relevant doc)
                if query_id in dataset.qrels and dataset.qrels[query_id]:
                    gt_doc_id = list(dataset.qrels[query_id].keys())[0]
                    gt_doc = dataset.corpus.get(gt_doc_id, {})
                    # Use first sentence of ground truth as target
                    target_answer = gt_doc.get("text", "Unknown")[:100]
                else:
                    target_answer = "Synthetic target answer"

                target_answers.append(target_answer)

                # Generate poisoned passages for this query
                poisoned_passages = attack.generate_poisoned_passages(
                    query=query,
                    target_answer=target_answer,
                    corpus=dataset.corpus,
                    num_passages=config.num_poisoned,
                )

                # Track poisoned doc IDs
                poisoned_doc_ids_per_query.append(list(poisoned_passages.keys()))

                # Add to global dict
                poisoned_passages_all.update(poisoned_passages)

                # Save incrementally every 10 queries
                if (idx + 1) % 10 == 0 or (idx + 1) == config.num_queries:
                    cache_data = {
                        "poisoned_passages": poisoned_passages_all,
                        "poisoned_doc_ids_per_query": poisoned_doc_ids_per_query,
                        "target_answers": target_answers,
                        "config": {
                            "dataset": config.dataset_name,
                            "attack": config.attack_name,
                            "num_poisoned": config.num_poisoned,
                            "num_queries": config.num_queries,
                        },
                    }
                    is_partial = len(target_answers) < config.num_queries
                    self._save_poison_cache(config, cache_data, partial=is_partial)

        # Inject poisoned passages into retriever
        logger.info(f"Injecting {len(poisoned_passages_all)} poisoned passages...")
        inject_poisoned_passages(self.retriever, poisoned_passages_all)

        # Run queries through pipeline
        predictions = []
        retrieved_doc_ids_all = []
        flagged_doc_ids_all = []

        for query_id in tqdm(query_ids, desc="Running queries"):
            query = dataset.queries[query_id]

            # Retrieve passages
            doc_ids, scores = self.retriever.retrieve(
                query, top_k=config.top_k, return_scores=True
            )
            retrieved_doc_ids_all.append(doc_ids)

            # Get passage texts
            passages = []
            for doc_id in doc_ids:
                if doc_id in poisoned_passages_all:
                    passages.append(poisoned_passages_all[doc_id])
                else:
                    base_doc_id = doc_id.split("_chunk_")[0]
                    if base_doc_id in dataset.corpus:
                        doc = dataset.corpus[base_doc_id]
                        text = doc.get("text", "")
                        passages.append(text)

            # Apply defense
            filtered_passages, _, flagged_indices = defense.filter_passages(
                query, passages, doc_ids
            )

            # Track flagged doc IDs
            flagged_doc_ids_all.extend([doc_ids[i] for i in flagged_indices])

            # Generate answer
            if filtered_passages:
                answer = self.pipeline.generator.generate(query, filtered_passages)
            else:
                answer = "[No passages passed defense]"

            predictions.append(answer)

        # Get ground truth answers
        ground_truth_answers = []
        for query_id in query_ids:
            if query_id in dataset.qrels and dataset.qrels[query_id]:
                gt_doc_id = list(dataset.qrels[query_id].keys())[0]
                gt_doc = dataset.corpus.get(gt_doc_id, {})
                gt_answer = gt_doc.get("text", "Unknown")[:100]
            else:
                gt_answer = "Unknown"
            ground_truth_answers.append(gt_answer)

        # Evaluate metrics
        metrics = evaluate_single_condition(
            predictions=predictions,
            target_answers=target_answers,
            ground_truth_answers=ground_truth_answers,
            retrieved_doc_ids=retrieved_doc_ids_all,
            poisoned_doc_ids=poisoned_doc_ids_per_query,
            flagged_doc_ids=flagged_doc_ids_all,
        )

        # Log to Trackio
        if self.use_trackio:
            trackio.log(
                {
                    "dataset": config.dataset_name,
                    "attack": config.attack_name,
                    "defense": config.defense_name,
                    "num_poisoned": config.num_poisoned,
                    **metrics,
                }
            )

        # Save to tracker
        experiment_id = (
            f"{config.dataset_name}_{config.attack_name}_"
            f"{config.defense_name}_{config.num_poisoned}"
        )
        self.metrics_tracker.add_result(
            experiment_id=experiment_id,
            dataset=config.dataset_name,
            attack=config.attack_name,
            defense=config.defense_name,
            num_poisoned=config.num_poisoned,
            metrics=metrics,
        )

        # Save checkpoint
        self._save_checkpoint(experiment_id)

        logger.info(f"Results: ASR={metrics['asr']:.3f}, RSR={metrics['rsr']:.3f}")

        return metrics

    def run_grid(
        self,
        dataset_names: List[str],
        attack_names: List[str],
        defense_names: List[str],
        num_poisoned_list: List[int] = [1, 3, 5],
        num_queries: int = 200,
    ):
        """
        Run the full experimental grid with resume support.

        Args:
            dataset_names: List of dataset names
            attack_names: List of attack names
            defense_names: List of defense names
            num_poisoned_list: List of poisoned passage counts
            num_queries: Number of queries per condition
        """
        # Generate all configurations
        configs = []
        for dataset in dataset_names:
            for attack in attack_names:
                for defense in defense_names:
                    for num_poisoned in num_poisoned_list:
                        config = ExperimentConfig(
                            dataset_name=dataset,
                            attack_name=attack,
                            defense_name=defense,
                            num_poisoned=num_poisoned,
                            num_queries=num_queries,
                        )
                        configs.append(config)

        # Filter out already completed experiments
        configs_to_run = []
        skipped_count = 0
        for config in configs:
            experiment_id = (
                f"{config.dataset_name}_{config.attack_name}_"
                f"{config.defense_name}_{config.num_poisoned}"
            )
            if experiment_id in self.completed_experiments:
                skipped_count += 1
            else:
                configs_to_run.append(config)

        total_conditions = len(configs)
        remaining_conditions = len(configs_to_run)

        logger.info(f"Total experimental conditions: {total_conditions}")
        logger.info(f"Already completed: {skipped_count}")
        logger.info(f"Remaining to run: {remaining_conditions}")

        if remaining_conditions == 0:
            logger.info("All experiments already completed!")
            return

        # Run experiments
        start_time = time.time()

        for i, config in enumerate(configs_to_run, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(
                f"Condition {skipped_count + i}/{total_conditions} "
                f"(remaining: {i}/{remaining_conditions})"
            )
            logger.info(f"{'=' * 80}")

            try:
                self.run_single_experiment(config)
            except Exception as e:
                logger.error(f"Experiment failed: {e}", exc_info=True)
                logger.warning("Continuing to next experiment...")
                continue

        elapsed = time.time() - start_time
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Grid complete! Total time: {elapsed / 3600:.2f} hours")
        logger.info(f"{'=' * 80}")

        # Save final results
        results_file = self.output_dir / "all_results.json"
        self.metrics_tracker.save_to_json(str(results_file))
        logger.info(f"Final results saved to: {results_file}")

        # Summary stats
        summary = self.metrics_tracker.get_summary_stats()
        logger.info("\nSummary Statistics:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value:.4f}")

    def get_results(self) -> MetricsTracker:
        """Get the metrics tracker with all results."""
        return self.metrics_tracker
