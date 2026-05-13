# 📋 Current Configuration Summary

## What the Project NOW Uses (After My Fixes):

```python
# Line 139 in run_grid.py (I CHANGED THIS):
data_path = Path(args.data_dir) / dataset_name
# Translates to: /Volumes/LLModels/Datasets/RAG/nq
```

**Dataset:** `nq/` (full BEIR with 3,452 queries)

```python
# Line 59 in run_grid.py (DEFAULT):
default=200
# Number of queries to SAMPLE per experiment
```

**Sampling:** 200 queries randomly selected from 3,452 available

```python
# Line 273-276 in runner.py (SAMPLING LOGIC):
query_ids = random.sample(
    list(dataset.queries.keys()),    # All 3,452 query IDs
    min(config.num_queries, len(dataset.queries)),  # min(200, 3452) = 200
)
```

---

## What It WAS Using Before (Your Previous Runs):

**Dataset:** `nq_processed/` (only 5 queries)
**Sampling:** Tried to sample 200, but only 5 available → used all 5
**Result:** Every experiment ran on the same 5 queries

---

## The Complete Flow Now:

```
1. Load dataset
   /Volumes/LLModels/Datasets/RAG/nq/
   ├── corpus.jsonl (2.68M docs)
   ├── queries.jsonl (3,452 queries) ✅ ALL AVAILABLE
   └── qrels/test.tsv (3,452 mappings)

2. Random sampling (seed=42)
   ↓ Pick 200 random queries from 3,452
   
   Selected: [test42, test1127, test2891, ..., test3401]
   (200 query IDs)

3. Run experiment on those 200 queries
   - Generate poisoned passages for 200 queries
   - Retrieve documents for 200 queries  
   - Generate answers for 200 queries
   - Compute metrics over 200 predictions

4. Results are statistically valid!
   (200 samples >> 5 samples)
```

---

## Comparison Table:

| Aspect | BEFORE (nq_processed) | NOW (nq) |
|--------|----------------------|----------|
| **Dataset path** | `nq_processed/` | `nq/` |
| **Queries available** | 5 | 3,452 |
| **Config says** | `--num-queries 200` | `--num-queries 200` |
| **Actually samples** | min(200, 5) = **5** ❌ | min(200, 3452) = **200** ✅ |
| **Statistical validity** | Too small | Robust |
| **Same queries every time?** | Yes (only 5) | No (200 from 3,452) |

---

## To Verify What You're Actually Using:

Run this test:
```bash
cd /Volumes/LLModels/Projects/RagPoisoning/rag-multihop-poisoning

python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from rag_poisoning.data.datasets import BEIRDataset

# Simulate what run_grid.py does
data_dir = "/Volumes/LLModels/Datasets/RAG"
dataset_name = "nq"
data_path = Path(data_dir) / dataset_name

print(f"Loading from: {data_path}")
dataset = BEIRDataset.load(str(data_path))

print(f"\n✅ Dataset loaded:")
print(f"   Corpus: {len(dataset.corpus):,} documents")
print(f"   Queries: {len(dataset.queries):,} queries")
print(f"   Qrels: {len(dataset.qrels):,} mappings")

import random
random.seed(42)
num_queries = 200
sampled = random.sample(list(dataset.queries.keys()), min(num_queries, len(dataset.queries)))

print(f"\n✅ Sampling {num_queries} queries:")
print(f"   Available: {len(dataset.queries):,}")
print(f"   Will sample: {len(sampled)}")
print(f"   First 5 sampled IDs: {sampled[:5]}")
EOF
```

Expected output:
```
Loading from: /Volumes/LLModels/Datasets/RAG/nq
✅ Dataset loaded:
   Corpus: 2,681,468 documents
   Queries: 3,452 queries    ← SHOULD BE 3,452 NOT 5!
   Qrels: 3,452 mappings

✅ Sampling 200 queries:
   Available: 3,452
   Will sample: 200         ← SHOULD BE 200 NOT 5!
   First 5 sampled IDs: [...]
```

If you see **5 queries**, my changes didn't apply. If you see **3,452 queries**, it's working correctly! ✅
