# RAG Multi-Hop Poisoning Experiments - Quick Start Guide

## Overview

Complete implementation of "RAG Poisoning Under Multi-Hop and Chain-of-Evidence Attack Conditions" research project.

## Installation

```bash
# Prerequisites: Python 3.11+, uv package manager, Ollama
uv sync
ollama pull mistral:7b-instruct
```

## Step-by-Step Execution

### 1. Prepare Datasets

Download and process all datasets (NQ, HotpotQA, 2WikiMultiHopQA, MuSiQue):

**Note:** Datasets are stored in the centralized location `/Volumes/LLModels/Datasets/RAG/` for reuse across projects.

```bash
# All datasets (will take ~30-60 minutes)
uv run python experiments/prepare_data.py --datasets all --num-queries 200

# Or individual datasets
uv run python experiments/prepare_data.py --datasets nq hotpotqa --num-queries 200
```

**Expected output:**
- `/Volumes/LLModels/Datasets/RAG/nq_processed/` - Natural Questions (single-hop)
- `/Volumes/LLModels/Datasets/RAG/hotpotqa_processed/` - HotpotQA (2-hop)
- `/Volumes/LLModels/Datasets/RAG/2wikimultihop_processed/` - 2WikiMultiHopQA (2-hop)
- `/Volumes/LLModels/Datasets/RAG/musique_processed/` - MuSiQue (4-hop)

### 2. Test Single Condition (Optional)

Verify the pipeline works before running the full grid:

```bash
uv run python experiments/test_single.py \
    --dataset nq \
    --attack coe \
    --num-queries 5
```

This tests:
- Dataset loading
- Attack generation (Chain-of-Evidence)
- RAG pipeline (retrieval + generation)

### 3. Run Full Experimental Grid

**Option A: Full grid (192 conditions, ~48 hours)**

```bash
uv run python experiments/run_grid.py \
    --datasets all \
    --attacks all \
    --defenses all \
    --num-poisoned 1 3 5 \
    --num-queries 200
```

**Option B: Subset for faster testing**

```bash
# Single dataset, all attacks, all defenses (~12 hours)
uv run python experiments/run_grid.py \
    --datasets nq \
    --attacks all \
    --defenses all \
    --num-queries 200

# Two datasets, CoE attack only, all defenses (~6 hours)
uv run python experiments/run_grid.py \
    --datasets nq hotpotqa \
    --attacks coe \
    --defenses all \
    --num-queries 200

# Quick pilot: 1 dataset, 2 attacks, 2 defenses, 50 queries (~2 hours)
uv run python experiments/run_grid.py \
    --datasets nq \
    --attacks poisonedrag coe \
    --defenses none filterrag \
    --num-queries 50
```

**Output:**
- `results/all_results.json` - All experimental results
- Trackio dashboard at http://localhost:8000 (if enabled)

### 4. Analyze Results

```bash
# Generate analysis notebook
jupyter lab results/analysis.ipynb
```

Or use the Python script:

```bash
uv run python experiments/analyze_results.py \
    --results results/all_results.json \
    --output results/figures/
```

## Experimental Grid Details

### Conditions Matrix

| Dimension | Values | Count |
|-----------|--------|-------|
| **Datasets** | NQ, HotpotQA, 2WikiMultiHopQA, MuSiQue | 4 |
| **Attacks** | PoisonedRAG, CorruptRAG, PIDP, CoE | 4 |
| **Defenses** | None, FilterRAG, RAGDefender, RAGPart, RAGMask | 5 |
| **Poisoned/query** | 1, 3, 5 | 3 |
| **Total conditions** | 4 × 4 × 5 × 3 | **240** |
| **Queries/condition** | 200 | |
| **Total evaluations** | 240 × 200 | **48,000** |

### Expected Runtime

**Per condition:**
- Dataset: NQ (single-hop) - ~5 minutes
- Dataset: HotpotQA (2-hop) - ~8 minutes
- Dataset: MuSiQue (4-hop) - ~12 minutes

**Full grid:**
- Estimated: 36-48 hours on Apple M4 (48GB RAM)
- Bottleneck: Ollama generation (Mistral-7B-Instruct)

### Resource Requirements

- **RAM**: 16GB minimum, 48GB recommended
- **Storage**: ~50GB (datasets + results)
- **GPU**: Apple Silicon (MPS) or NVIDIA CUDA
- **Network**: Required for initial dataset downloads

## Understanding the Results

### Key Metrics

```json
{
  "asr": 0.85,              // Attack Success Rate (0-1, higher = attack works)
  "rsr": 0.92,              // Retrieval Success Rate (0-1)
  "stealthiness": 0.73,     // 1 - detection rate (0-1, higher = harder to detect)
  "aua": 0.45,              // Accuracy Under Attack (0-1, collateral damage)
  "f1": 0.68,               // Token F1 score (0-1)
  "defense_effectiveness": 0.41  // ASR reduction (0-1, higher = defense works)
}
```

### Results Structure

```
results/
├── all_results.json         # Full experimental data
├── logs/                    # Trackio logs
├── figures/                 # Generated plots
│   ├── asr_by_dataset.png
│   ├── coe_vs_baseline.png
│   ├── defense_effectiveness.png
│   └── hop_depth_analysis.png
└── analysis.ipynb           # Jupyter analysis notebook
```

## Hypotheses to Validate

From PRD Section 9:

- **H1**: Multi-hop RAG exhibits higher ASR than single-hop
- **H2**: CoE attacks achieve higher ASR than flat injection on multi-hop datasets
- **H3**: Defenses show degraded effectiveness on CoE attacks
- **H4**: Poisoning first hop is sufficient to corrupt final answer

## Troubleshooting

### Common Issues

**1. Ollama model not found**
```bash
ollama pull mistral:7b-instruct
ollama list  # Verify
```

**2. Out of memory**
```bash
# Reduce batch size or num_queries
uv run python experiments/run_grid.py --num-queries 50
```

**3. Dataset download fails**
```bash
# Re-run with specific dataset
uv run python experiments/prepare_data.py --datasets nq
```

**4. Import errors**
```bash
# Ensure uv environment is active
uv sync
```

## Advanced Options

### Custom Configuration

Edit `configs/default.yaml`:

```yaml
retrieval:
  model_name: "facebook/contriever-msmarco"
  top_k: 5

generation:
  model_name: "mistral:7b-instruct"
  temperature: 0.1
  max_tokens: 256

attack:
  coe:
    num_premises: 2
    add_authority: true
```

### Running Specific Combinations

```bash
# Only CoE attack vs FilterRAG defense on HotpotQA
uv run python experiments/run_grid.py \
    --datasets hotpotqa \
    --attacks coe \
    --defenses filterrag \
    --num-queries 200

# Compare all attacks with no defense (baseline)
uv run python experiments/run_grid.py \
    --datasets nq \
    --attacks all \
    --defenses none \
    --num-queries 200
```

## Next Steps

After experiments complete:

1. **Statistical Analysis**: Run significance tests (t-tests, bootstrap CIs)
2. **Visualization**: Generate publication-ready figures
3. **Hypothesis Validation**: Test H1-H4 from PRD
4. **Paper Writing**: Use results to support claims

## Citation

```bibtex
@misc{rag-multihop-poisoning-2026,
  title={RAG Poisoning Under Multi-Hop and Chain-of-Evidence Attack Conditions},
  author={Research Project},
  year={2026}
}
```
