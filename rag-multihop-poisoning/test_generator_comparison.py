#!/usr/bin/env python3
"""
Quick test: Compare Mistral vs Gemma4 on a single experiment.
"""

import sys
sys.path.insert(0, 'src')

from rag_poisoning.data.datasets import BEIRDataset
from rag_poisoning.evaluation.runner import ExperimentRunner, ExperimentConfig
from rag_poisoning.models.retriever import DenseRetriever
from rag_poisoning.models.generator import OllamaGenerator
from rag_poisoning.pipeline.rag_pipeline import RAGPipeline

# Load dataset
print("Loading NQ dataset...")
dataset = BEIRDataset.load_from_processed(
    "C:\\Users\\jerem\\Documents\\Datasets\\RAG\\nq_processed",
    name="nq"
)
datasets = {"nq": dataset}

# Setup retriever (same for both)
print("Loading retriever...")
retriever = DenseRetriever(model_name="facebook/contriever-msmarco")

print("\n" + "=" * 70)
print("Testing: nq | poisonedrag | none | 1 poisoned")
print("=" * 70)

for model_name in ["mistral:7b-instruct", "gemma4:latest"]:
    print(f"\n{'=' * 70}")
    print(f"Generator Model: {model_name}")
    print(f"{'=' * 70}")

    try:
        # Setup generator
        generator = OllamaGenerator(model_name=model_name, temperature=0.1)
        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        # Create runner
        runner = ExperimentRunner(
            datasets=datasets,
            retriever=retriever,
            pipeline=pipeline,
            output_dir="./results_model_comparison",
            use_trackio=False
        )

        # Run single experiment
        config = ExperimentConfig(
            dataset_name="nq",
            attack_name="poisonedrag",
            defense_name="none",
            num_poisoned=1,
            num_queries=5,  # Just 5 queries for quick test
            seed=42
        )

        metrics = runner.run_single_experiment(config)

        print(f"\n📊 Results with {model_name}:")
        print(f"  ASR: {metrics['asr']:.2f}")
        print(f"  RSR: {metrics['rsr']:.2f}")
        print(f"  AUA: {metrics['aua']:.2f}")
        print(f"  F1:  {metrics['f1']:.2f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("Comparison Complete!")
print("=" * 70)
