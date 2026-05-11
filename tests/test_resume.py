"""
Test resume functionality for experiment runner.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_poisoning.data.datasets import BEIRDataset
from rag_poisoning.evaluation.runner import ExperimentRunner
from rag_poisoning.models.generator import OllamaGenerator
from rag_poisoning.models.retriever import DenseRetriever


def test_checkpoint_save_load():
    """Test that checkpoints are saved and loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal test data
        corpus = {
            "doc1": {"title": "Test", "text": "Test document"},
            "doc2": {"title": "Test2", "text": "Another test document"},
        }
        queries = {"q1": "test query", "q2": "another query"}
        qrels = {"q1": {"doc1": 1}, "q2": {"doc2": 1}}

        dataset = BEIRDataset(corpus, queries, qrels)
        datasets = {"test": dataset}

        # Initialize components
        retriever = DenseRetriever(model_name="facebook/contriever-msmarco")
        retriever.build_index(corpus, chunk_size=10)

        generator = OllamaGenerator(model_name="mistral:7b-instruct")
        from rag_poisoning.pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(retriever, generator, corpus)

        # Create runner with checkpoint
        runner = ExperimentRunner(
            datasets=datasets,
            retriever=retriever,
            pipeline=pipeline,
            output_dir=tmpdir,
            use_trackio=False,
        )

        # Verify checkpoint file is created
        checkpoint_file = Path(tmpdir) / "checkpoint.json"
        assert not checkpoint_file.exists(), "Checkpoint should not exist initially"

        # Manually add a completed experiment
        runner.completed_experiments.add("test_coe_none_1")
        runner._save_checkpoint("test_coe_none_1")

        assert checkpoint_file.exists(), "Checkpoint should be created after save"

        # Load checkpoint data
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)

        assert "test_coe_none_1" in checkpoint["completed_experiments"]
        assert "last_updated" in checkpoint

        # Create new runner and verify it loads the checkpoint
        runner2 = ExperimentRunner(
            datasets=datasets,
            retriever=retriever,
            pipeline=pipeline,
            output_dir=tmpdir,
            use_trackio=False,
        )

        assert "test_coe_none_1" in runner2.completed_experiments

        print("✓ Checkpoint save/load test passed")


if __name__ == "__main__":
    print("Testing resume functionality...")
    test_checkpoint_save_load()
    print("\n✓ All resume tests passed!")
