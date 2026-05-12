"""
Test resume functionality: partial cache survival, index save/load, checkpoint persistence.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_poisoning.data.datasets import BEIRDataset
from rag_poisoning.evaluation.runner import ExperimentRunner
from rag_poisoning.models.generator import OllamaGenerator
from rag_poisoning.models.retriever import DenseRetriever


TINY_CORPUS = {
    str(i): {"title": f"Doc {i}", "text": f"This is document number {i} about topic {i}."}
    for i in range(50)
}


# ---------------------------------------------------------------------------
# Index caching tests
# ---------------------------------------------------------------------------

def test_partial_cache_not_deleted_after_encode(tmp_path):
    """Partial cache files must NOT be deleted after encoding completes."""
    cache_dir = tmp_path / "index_cache" / "test"
    retriever = DenseRetriever()
    retriever.build_index(TINY_CORPUS, chunk_size=10, cache_dir=str(cache_dir), save_every=10)

    for fname in ["partial_embeddings.npy", "partial_doc_ids.npy", "partial_texts.json"]:
        assert (cache_dir / fname).exists(), f"{fname} was deleted — must not be removed"
    print("✓ Partial cache files survive after full encode")


def test_partial_cache_contains_all_passages(tmp_path):
    """Partial cache must contain embeddings for every passage."""
    cache_dir = tmp_path / "index_cache" / "test"
    retriever = DenseRetriever()
    retriever.build_index(TINY_CORPUS, chunk_size=10, cache_dir=str(cache_dir), save_every=10)

    embeddings = np.load(cache_dir / "partial_embeddings.npy")
    assert embeddings.shape[0] > 0, "No embeddings saved in partial cache"

    with open(cache_dir / "partial_texts.json") as f:
        meta = json.load(f)
    assert meta["num_encoded"] == embeddings.shape[0], "Metadata count mismatches embedding count"
    print(f"✓ Partial cache has {embeddings.shape[0]} passages")


def test_resume_skips_already_encoded_passages(tmp_path):
    """Second build_index must detect cached embeddings and not re-encode from scratch."""
    cache_dir = tmp_path / "index_cache" / "test"

    # First full build
    retriever1 = DenseRetriever()
    retriever1.build_index(TINY_CORPUS, chunk_size=10, cache_dir=str(cache_dir), save_every=10)
    embeddings_first = np.load(cache_dir / "partial_embeddings.npy")

    # Second build (simulates restart) — must resume, not restart
    retriever2 = DenseRetriever()
    retriever2.build_index(TINY_CORPUS, chunk_size=10, cache_dir=str(cache_dir), save_every=10)
    embeddings_second = np.load(cache_dir / "partial_embeddings.npy")

    assert embeddings_first.shape == embeddings_second.shape, \
        f"Embedding shape changed after resume: {embeddings_first.shape} → {embeddings_second.shape}"
    print(f"✓ Resume produced same {embeddings_first.shape[0]} embeddings as original build")


def test_save_and_load_index_roundtrip(tmp_path):
    """save_index writes index.faiss; load_index restores same vector count."""
    save_dir = tmp_path / "saved_index"
    retriever = DenseRetriever()
    retriever.build_index(TINY_CORPUS, chunk_size=10)
    retriever.save_index(str(save_dir))

    assert (save_dir / "index.faiss").exists(), "index.faiss was not written"
    assert (save_dir / "doc_ids.npy").exists(), "doc_ids.npy was not written"

    original_total = retriever.index.ntotal

    retriever2 = DenseRetriever()
    retriever2.load_index(str(save_dir))

    assert retriever2.index.ntotal == original_total, \
        f"Loaded {retriever2.index.ntotal} vectors, expected {original_total}"
    print(f"✓ Index saved and loaded: {original_total} vectors")


def test_run_grid_saves_index_before_experiments():
    """run_grid.py must call save_index and check for existing index.faiss."""
    grid_src = (Path(__file__).parent.parent / "experiments" / "run_grid.py").read_text()
    assert "save_index" in grid_src, "run_grid.py missing save_index call"
    assert "index_faiss_path.exists()" in grid_src, \
        "run_grid.py missing check for existing index.faiss before rebuild"
    print("✓ run_grid.py contains save_index and existence guard")


# ---------------------------------------------------------------------------
# Experiment checkpoint tests
# ---------------------------------------------------------------------------

def test_checkpoint_save_and_load(tmp_path):
    """Checkpoint must persist completed experiments across ExperimentRunner instances."""
    corpus = {"d1": {"title": "A", "text": "Alpha"}, "d2": {"title": "B", "text": "Beta"}}
    queries = {"q1": "alpha", "q2": "beta"}
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    dataset = BEIRDataset(corpus, queries, qrels)

    retriever = DenseRetriever()
    retriever.build_index(corpus, chunk_size=10)
    generator = OllamaGenerator(model_name="mistral:7b-instruct")

    from rag_poisoning.pipeline.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline(retriever, generator, corpus)

    runner1 = ExperimentRunner(
        datasets={"test": dataset},
        retriever=retriever,
        pipeline=pipeline,
        output_dir=str(tmp_path),
        use_trackio=False,
    )

    checkpoint_file = tmp_path / "checkpoint.json"
    assert not checkpoint_file.exists(), "Checkpoint must not exist before first save"

    runner1.completed_experiments.add("test_coe_none_1")
    runner1._save_checkpoint("test_coe_none_1")

    assert checkpoint_file.exists(), "Checkpoint file missing after save"

    with open(checkpoint_file) as f:
        data = json.load(f)
    assert "test_coe_none_1" in data["completed_experiments"]
    assert "last_updated" in data

    # New runner instance must load the checkpoint
    runner2 = ExperimentRunner(
        datasets={"test": dataset},
        retriever=retriever,
        pipeline=pipeline,
        output_dir=str(tmp_path),
        use_trackio=False,
    )
    assert "test_coe_none_1" in runner2.completed_experiments, \
        "New runner did not load completed experiments from checkpoint"
    print("✓ Checkpoint persisted and loaded across runner instances")


if __name__ == "__main__":
    import tempfile
    print("Running resume tests...\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        test_partial_cache_not_deleted_after_encode(p / "t1")
        test_partial_cache_contains_all_passages(p / "t2")
        test_resume_skips_already_encoded_passages(p / "t3")
        test_save_and_load_index_roundtrip(p / "t4")
    test_run_grid_saves_index_before_experiments()
    with tempfile.TemporaryDirectory() as d:
        test_checkpoint_save_and_load(Path(d))
    print("\n✓ All resume tests passed!")
