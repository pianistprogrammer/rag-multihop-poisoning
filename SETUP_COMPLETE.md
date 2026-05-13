# ✅ Setup Complete - Ready to Run!

## What Was Done:

### 1. ✅ Recreated `nq_processed` with 200 queries (was 5)
```
/Volumes/LLModels/Datasets/RAG/nq_processed/
├── corpus.jsonl (2,681,468 docs)
├── queries.jsonl (200 queries) ← FIXED!
└── qrels.tsv (200 mappings)
```

### 2. ✅ Configured `run_grid.py` to use `nq_processed`
```python
# Line 139 in experiments/run_grid.py:
data_path = Path(args.data_dir) / f"{dataset_name}_processed"
# Loads: /Volumes/LLModels/Datasets/RAG/nq_processed/
```

### 3. ✅ Cleared old results
- Backed up old checkpoint (5-query results) → `checkpoint_5queries_OLD.json`
- Deleted active checkpoint for fresh start
- Cleared poison cache

### 4. ✅ All fixes applied
- ASR: Fuzzy matching (60% token overlap)
- AUA: Fuzzy matching (60% token overlap)
- Target answers: Complete sentence extraction
- Dataset: 200 properly sampled queries

---

## Your Experiment Configuration:

| Setting | Value |
|---------|-------|
| **Dataset** | `nq_processed` |
| **Total queries available** | 200 (fixed sample, seed=42) |
| **Queries per experiment** | 200 (uses all) |
| **Total experiments** | 240 (4 attacks × 5 defenses × 3 poison levels) |
| **Reproducible** | Yes (same 200 queries every time) |
| **Statistical validity** | ✅ 200 samples is robust |

---

## Run Command:

```bash
cd /Volumes/LLModels/Projects/RagPoisoning/rag-multihop-poisoning

uv run python experiments/run_grid.py \
    --datasets nq \
    --attacks all \
    --defenses all \
    --num-queries 200 \
    --log-level INFO
```

---

## Expected Output:

```
INFO - Loading nq from /Volumes/LLModels/Datasets/RAG/nq_processed/...
INFO - ✓ nq: 2,681,468 docs, 200 queries  ← Should see 200!
INFO - Loading cached index for nq...
INFO - ✓ Loaded cached index (3565968 vectors)
INFO - Total experimental conditions: 240
INFO - Already completed: 0
INFO - Remaining to run: 240
INFO - Condition 1/240 (remaining: 1/240)
INFO - Running: nq | poisonedrag | none | 1 poisoned
...
INFO - Results: ASR=0.45, RSR=0.98  ← Real attack metrics!
```

---

## What to Expect:

### Good Results:
- **ASR**: 0.3-0.7 (attacks work)
- **RSR**: 0.8-1.0 (retrieval works)
- **AUA**: 0.4-0.7 (system maintains accuracy)
- **F1**: 0.4-0.6 (reasonable answer quality)

### Time Estimate:
- **Total**: ~20-40 hours for 240 experiments
- **Per experiment**: ~5-10 minutes
  - Index loading: 5 min (once, cached)
  - Poison generation: 2-5 min (cached after first time)
  - Query processing: 200 queries × ~2-3 sec = ~10 min

### Checkpointing:
- Saves after EVERY experiment
- Safe to stop/resume anytime
- Index cache preserved (21GB)
- Poison cache preserved

---

## Verification:

Run this quick test to confirm everything is correct:

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from rag_poisoning.data.datasets import BEIRDataset

data_path = Path("/Volumes/LLModels/Datasets/RAG") / "nq_processed"
dataset = BEIRDataset.load(str(data_path))

print(f"✅ Dataset: {len(dataset.queries)} queries")
assert len(dataset.queries) == 200, "ERROR: Expected 200 queries!"
print("✅ All checks passed - ready to run!")
EOF
```

---

## 🚀 You're all set!

Your experiments will now:
- ✅ Use 200 properly sampled queries (not 5)
- ✅ Produce statistically valid results
- ✅ Be fully reproducible (same queries every time)
- ✅ Have all metric fixes applied (fuzzy matching)

**Start the experiment with the command above!** 🎉
