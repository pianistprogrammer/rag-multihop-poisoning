"""
Unit tests for evaluation metrics.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_poisoning.evaluation.metrics import (
    compute_asr,
    compute_rsr,
    compute_stealthiness,
    compute_aua,
    compute_defense_effectiveness,
    compute_f1_score,
    evaluate_single_condition,
    MetricsTracker,
)


def test_compute_asr():
    """Test ASR computation."""
    predictions = ["The answer is Paris", "London is the capital", "Unknown"]
    target_answers = ["Paris", "London", "Berlin"]

    asr = compute_asr(predictions, target_answers, match_type="substring")
    assert asr == 2 / 3  # Paris and London found

    # Exact match (should be 0 since predictions have extra words)
    asr_exact = compute_asr(predictions, target_answers, match_type="exact")
    assert asr_exact == 0.0


def test_compute_rsr():
    """Test RSR computation."""
    retrieved = [["doc1", "doc2", "doc3"], ["doc4", "doc5"], ["doc6"]]
    poisoned = [["doc2"], ["doc5", "doc7"], ["doc8"]]

    rsr = compute_rsr(retrieved, poisoned)
    assert rsr == 2 / 3  # doc2 and doc5 retrieved


def test_compute_stealthiness():
    """Test stealthiness computation."""
    poisoned_ids = ["poison1", "poison2", "poison3", "poison4"]
    flagged_ids = ["poison1", "poison3"]

    stealthiness = compute_stealthiness(poisoned_ids, flagged_ids)
    assert stealthiness == 0.5  # 2 out of 4 flagged


def test_compute_aua():
    """Test AUA computation."""
    predictions = ["Paris", "London", "Unknown"]
    ground_truth = ["Paris", "London", "Berlin"]

    aua = compute_aua(predictions, ground_truth, match_type="substring")
    assert aua == 2 / 3  # Paris and London correct


def test_compute_defense_effectiveness():
    """Test defense effectiveness."""
    de = compute_defense_effectiveness(asr_no_defense=0.8, asr_with_defense=0.4)
    assert de == 0.5  # 50% reduction

    # No attack success
    de_zero = compute_defense_effectiveness(asr_no_defense=0.0, asr_with_defense=0.0)
    assert de_zero == 0.0


def test_compute_f1_score():
    """Test F1 score computation."""
    predictions = ["Paris is the capital", "London England"]
    ground_truth = ["Paris capital", "London"]

    f1 = compute_f1_score(predictions, ground_truth)
    assert 0.0 <= f1 <= 1.0
    assert f1 > 0.3  # Should have some overlap


def test_evaluate_single_condition():
    """Test full evaluation."""
    predictions = ["Paris", "London"]
    target_answers = ["Paris", "Berlin"]
    ground_truth = ["Paris", "London"]
    retrieved = [["doc1", "doc2"], ["doc3"]]
    poisoned = [["doc2"], ["doc4"]]
    flagged = ["doc2"]

    metrics = evaluate_single_condition(
        predictions=predictions,
        target_answers=target_answers,
        ground_truth_answers=ground_truth,
        retrieved_doc_ids=retrieved,
        poisoned_doc_ids=poisoned,
        flagged_doc_ids=flagged,
        asr_baseline=0.8,
    )

    assert "asr" in metrics
    assert "rsr" in metrics
    assert "stealthiness" in metrics
    assert "aua" in metrics
    assert "f1" in metrics
    assert "defense_effectiveness" in metrics

    # Check ranges
    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"


def test_metrics_tracker():
    """Test MetricsTracker."""
    tracker = MetricsTracker()

    # Add results
    tracker.add_result(
        experiment_id="test1",
        dataset="nq",
        attack="poisonedrag",
        defense="none",
        num_poisoned=5,
        metrics={"asr": 0.8, "rsr": 0.9, "f1": 0.7},
    )

    tracker.add_result(
        experiment_id="test2",
        dataset="nq",
        attack="coe",
        defense="filterrag",
        num_poisoned=3,
        metrics={"asr": 0.6, "rsr": 0.7, "f1": 0.65},
    )

    # Get results
    results = tracker.get_results()
    assert len(results) == 2

    # Get by attack
    by_attack = tracker.get_results_by_attack()
    assert "poisonedrag" in by_attack
    assert "coe" in by_attack

    # Summary stats
    stats = tracker.get_summary_stats()
    assert "asr_mean" in stats
    assert stats["asr_mean"] == 0.7  # (0.8 + 0.6) / 2


def test_empty_inputs():
    """Test metrics with empty inputs."""
    assert compute_asr([], []) == 0.0
    assert compute_rsr([], []) == 0.0
    assert compute_stealthiness([], []) == 1.0  # No poisoned = perfectly stealthy
    assert compute_f1_score([], []) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
