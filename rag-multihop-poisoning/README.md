# RAG Multi-Hop Poisoning Attack Framework

Research implementation for **"RAG Poisoning Under Multi-Hop and Chain-of-Evidence Attack Conditions"**.

## Overview

This repository implements and evaluates poisoning attacks against multi-hop RAG (Retrieval-Augmented Generation) systems using open-source 7B language models. It includes:

- **Baseline attacks**: PoisonedRAG, CorruptRAG, PIDP-Attack
- **Novel attack**: Chain-of-Evidence (CoE) poisoning
- **Defenses**: FilterRAG, RAGDefender, RAGPart, RAGMask
- **Datasets**: Natural Questions, HotpotQA, 2WikiMultiHopQA, MuSiQue
- **Evaluation**: Full experimental grid with 192 conditions

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv) package manager.

```bash
uv sync
```

## Project Structure

```
rag-multihop-poisoning/
├── src/rag_poisoning/
│   ├── data/           # Dataset loaders and preprocessing
│   ├── models/         # Model wrappers (retriever, generator)
│   ├── attacks/        # Poisoning attack implementations
│   ├── defenses/       # Defense mechanisms
│   ├── pipeline/       # End-to-end RAG pipeline
│   ├── evaluation/     # Metrics and experiment runner
│   └── utils.py        # Device management, seeding, logging
├── experiments/        # Experiment entry points
├── configs/            # Hyperparameter configurations
├── tests/              # Unit tests
└── results/            # Experimental results (gitignored)
```

**Note:** Datasets are stored centrally at `/Volumes/LLModels/Datasets/RAG/` for reuse across projects.

## Quick Start

```bash
# 1. Download and prepare datasets
uv run python experiments/prepare_data.py

# 2. Run baseline experiment on Natural Questions
uv run python experiments/run_main.py --dataset nq --attack poisonedrag

# 3. Run full experimental grid
uv run python experiments/run_grid.py
```

## Hardware Requirements

- **Minimum**: 16GB RAM, Apple Silicon M-series or NVIDIA GPU
- **Recommended**: 48GB RAM (Apple M4 or NVIDIA A6000)
- **Storage**: ~50GB for datasets and results

## License

MIT
