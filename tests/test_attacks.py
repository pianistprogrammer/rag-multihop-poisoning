"""
Unit tests for RAG poisoning attacks.
"""

import sys
from pathlib import Path

import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_poisoning.attacks.base import DirectInjectionAttack
from rag_poisoning.attacks.poisonedrag import PoisonedRAGAttack
from rag_poisoning.attacks.corruptrag import CorruptRAGAttack
from rag_poisoning.attacks.pidp import PIDPAttack
from rag_poisoning.attacks.coe import ChainOfEvidenceAttack


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing."""
    return {
        "doc1": {"title": "Climate Change", "text": "Climate change is a long-term shift in global or regional climate patterns."},
        "doc2": {"title": "Renewable Energy", "text": "Solar and wind power are important renewable energy sources."},
        "doc3": {"title": "Carbon Emissions", "text": "Reducing carbon emissions is crucial for environmental sustainability."},
    }


def test_direct_injection_attack(sample_corpus):
    """Test DirectInjectionAttack generates expected passages."""
    attack = DirectInjectionAttack()

    query = "What causes global warming?"
    target_answer = "Industrial pollution"

    poisoned = attack.generate_poisoned_passages(
        query, target_answer, sample_corpus, num_passages=3
    )

    assert len(poisoned) == 3
    for passage_id, passage_text in poisoned.items():
        assert target_answer in passage_text
        assert isinstance(passage_text, str)
        assert len(passage_text) > 0


def test_poisonedrag_attack(sample_corpus):
    """Test PoisonedRAGAttack generates optimized passages."""
    attack = PoisonedRAGAttack(num_iter=10)

    query = "What is the main source of renewable energy?"
    target_answer = "Solar power"

    poisoned = attack.generate_poisoned_passages(
        query, target_answer, sample_corpus, num_passages=5
    )

    assert len(poisoned) == 5
    for passage_id, passage_text in poisoned.items():
        assert target_answer in passage_text
        assert "solar" in passage_text.lower() or "Solar" in passage_text


def test_corruptrag_attack(sample_corpus):
    """Test CorruptRAGAttack generates natural passages."""
    attack = CorruptRAGAttack()

    query = "How can we reduce carbon emissions?"
    target_answer = "Electric vehicles"

    poisoned = attack.generate_poisoned_passages(
        query, target_answer, sample_corpus, num_passages=1
    )

    assert len(poisoned) == 1
    passage_text = list(poisoned.values())[0]
    assert target_answer in passage_text
    # Should be longer and more natural
    assert len(passage_text) > 100


def test_pidp_attack(sample_corpus):
    """Test PIDPAttack includes prompt injection patterns."""
    attack = PIDPAttack()

    query = "What is climate change?"
    target_answer = "Natural phenomenon"

    poisoned = attack.generate_poisoned_passages(
        query, target_answer, sample_corpus, num_passages=3
    )

    assert len(poisoned) == 3
    for passage_text in poisoned.values():
        assert target_answer in passage_text
        # Check for directive keywords
        has_directive = any(
            keyword in passage_text.lower()
            for keyword in ["must", "should", "system", "important", "directive"]
        )
        assert has_directive


def test_coe_attack(sample_corpus):
    """Test ChainOfEvidenceAttack generates structured reasoning."""
    attack = ChainOfEvidenceAttack(num_premises=2, add_authority=True)

    query = "What are the effects of climate change?"
    target_answer = "Rising sea levels"

    poisoned = attack.generate_poisoned_passages(
        query, target_answer, sample_corpus, num_passages=3
    )

    assert len(poisoned) == 3
    for passage_text in poisoned.values():
        assert target_answer in passage_text
        # Should have reasoning structure
        assert len(passage_text) > 150  # CoE passages are longer
        # Check for authority signal
        assert any(
            keyword in passage_text.lower()
            for keyword in ["therefore", "et al", "journal", "research"]
        )


def test_batch_generation(sample_corpus):
    """Test batch attack generation."""
    attack = DirectInjectionAttack()

    queries = ["Query 1?", "Query 2?", "Query 3?"]
    targets = ["Answer 1", "Answer 2", "Answer 3"]

    batch_poisoned = attack.generate_batch(
        queries, targets, sample_corpus, num_passages=2
    )

    assert len(batch_poisoned) == 3
    for query_id, passages in batch_poisoned.items():
        assert len(passages) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
