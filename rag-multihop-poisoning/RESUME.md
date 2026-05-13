# Resume Functionality

Complete guide to the 3-level checkpoint and caching system for fault-tolerant experiment execution.

## Overview

The experiment runner provides automatic checkpointing and resume functionality with three levels of protection against work loss:

1. **Level 1: Experiment-level checkpoints** - After each complete experiment
2. **Level 2: Incremental poison passage caching** - Saves every 10 queries during generation
3. **Level 3: FAISS index caching with incremental encoding** - Saves every 10,000 passages during index building

**Result**: You can safely interrupt at ANY time with minimal work loss (5-15 minutes max).

---

## Level 1: Experiment-Level Checkpoints

### How It Works

1. **Automatic Checkpointing**: After each experiment completes successfully, a checkpoint is saved to `results/checkpoint.json`
2. **Resume on Restart**: Re-running the experiment grid automatically detects and skips completed experiments
3. **Progress Preservation**: All metrics and results are preserved in the checkpoint file

### Checkpoint File Structure

```json
{
  "completed_experiments": [
    "nq_coe_none_1",
    "nq_coe_filterrag_1",
    ...
  ],
  "results": {
    "nq_coe_none_1": {
      "dataset": "nq",
      "attack": "coe",
      "defense": "none",
      "num_poisoned": 1,
      "metrics": {
        "asr": 0.85,
        "rsr": 0.12,
        ...
      }
    }
  },
  "last_updated": 1715434567.123
}
```

### What Gets Protected

- Completed experiments (200 queries + RAG pipeline per experiment)
- All computed metrics
- Experiment progress across the full 240-condition grid

### Maximum Loss

If interrupted during an experiment: Current experiment only (will restart from beginning of that experiment)

---

## Level 2: Incremental Poison Passage Caching

### How It Works

**The slow part** of each experiment is generating poisoned passages. This is now protected with incremental caching:

1. **Incremental Caching**: Poisoned passages are saved **every 10 queries** (not just at the end)
   - After query 10 → cache saved ✓
   - After query 20 → cache saved ✓
   - After query 30 → cache saved ✓
   - ... continues until all 200 queries done

2. **Automatic Resume**: If interrupted mid-generation:
   - Loads partial cache (e.g., 60 queries completed)
   - Skips already-completed queries
   - Continues from query 61
   - Keeps saving every 10 queries

3. **Cache Location**: `results/poison_cache/{dataset}_{attack}_{num_poisoned}_{num_queries}.json`

### Example Flow

```
Generating poisoned passages: 100%|████| 200/200
  └─ Query 10 → ✓ Cache saved (partial 10/200)
  └─ Query 20 → ✓ Cache saved (partial 20/200)
  └─ Query 30 → ✓ Cache saved (partial 30/200)
  └─ [INTERRUPT HERE - Ctrl+C]
  
Restart:
  └─ Load cache (30/200 queries)
  └─ Skip queries 1-30
  └─ Continue from query 31
  └─ Query 40 → ✓ Cache saved (partial 40/200)
  └─ ...
```

### What Gets Protected

- Generated poisoned passages for each query
- Target answers
- Poisoned document IDs per query

### Maximum Loss

At most 9 queries worth of work (~10-15 minutes depending on attack method)

---

## Level 3: FAISS Index Caching with Incremental Encoding

### How It Works

**The slowest part** of setup is building the FAISS index (encoding millions of documents). This now has two-level protection:

#### 3A: Incremental Encoding During Index Build

1. **Progress Saved Every 10,000 Passages**
   - Passage 10,000 → ✓ Partial cache saved
   - Passage 20,000 → ✓ Partial cache saved
   - ... continues until all passages encoded

2. **Automatic Resume from Partial Cache**
   - Check if partial cache exists (e.g., 50,000/3.5M passages done)
   - Load partial embeddings
   - Continue encoding from passage 50,001
   - Keep saving every 10,000 passages

3. **Partial Cache Files** (cleaned up after successful completion):
   - `results/index_cache/{dataset}/partial_embeddings.npy`
   - `results/index_cache/{dataset}/partial_doc_ids.npy`
   - `results/index_cache/{dataset}/partial_texts.json`

#### 3B: Final Index Caching

Complete index cached per dataset to `results/index_cache/{dataset_name}/`:
- `nq` index: ~3.5M passages, ~2-3 hours to build (first time only)
- `hotpotqa` index: ~400 passages, ~30 seconds
- `2wikimultihop` index: ~400 passages, ~30 seconds
- `musique` index: ~4K passages, ~2 minutes

### Example Flow (NQ Dataset - 3.5M Passages)

```
First attempt (interrupted):
  └─ Encoding: 10,000 passages → ✓ Partial saved
  └─ Encoding: 20,000 passages → ✓ Partial saved
  └─ Encoding: 30,000 passages → ✓ Partial saved
  └─ [INTERRUPT - Ctrl+C at 31%]

Resume (automatic):
  └─ Found partial cache: 30,000/3,565,968 passages
  └─ Loading cached embeddings...
  └─ Continuing from passage 30,001
  └─ Encoding: 40,000 passages → ✓ Partial saved
  └─ ...
  └─ All 3.5M encoded → ✓ Final index cached
  └─ Partial cache cleaned up

Future runs (any experiment using NQ):
  └─ Loading cached index... (2 seconds) ✓
```

### What Gets Protected

- Document embeddings (encoded passages)
- Document IDs mapping
- Complete FAISS index

### Maximum Loss

At most 9,999 passages worth of encoding (~5-10 minutes for large datasets like NQ)

---

## Usage

### Normal Usage (Automatic Resume)

Simply run the same command again after any interruption:

```bash
# First run (interrupted after 50 experiments)
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all

# Resume (automatically skips first 50, continues from #51)
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all
```

**Important**: Use the same parameters (datasets, attacks, defenses, num-poisoned) to ensure correct resume behavior.

### Commands

```bash
# Start experiments
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all

# Stop at any time
Ctrl+C

# Resume (same command)
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all
```

---

## Manual Checkpoint Management

### Check Progress

```bash
# View experiment checkpoint status
python -c "
import json
with open('results/checkpoint.json') as f:
    data = json.load(f)
    print(f'Completed experiments: {len(data[\"completed_experiments\"])}')
    print(f'Last updated: {data[\"last_updated\"]}')
"

# Check if poison cache exists for a config
ls -lh results/poison_cache/

# Check if index cache exists
ls -lh results/index_cache/
```

### Reset Everything

Start completely fresh (delete all caches and checkpoints):

```bash
rm -rf results/
```

### Clear Specific Caches

```bash
# Clear experiment checkpoints (re-run all experiments)
rm results/checkpoint.json results/all_results.json

# Clear poison cache (regenerate poisoned passages)
rm -rf results/poison_cache/

# Clear index cache (rebuild all indices)
rm -rf results/index_cache/
```

---

## Example Scenarios

### Scenario 1: Crash During Index Building

```bash
# Run starts, begins building NQ index
# 31% through encoding (30,000/111,437 batches)
# [CRASH - power failure]

# Resume: Same command
uv run python experiments/run_grid.py --datasets all ...

# Output:
# Building index for nq...
# Found partial cache: 30,000 passages encoded
# Continuing from passage 30,001...
# (resumes with ~5 min loss instead of 2h 50min)
```

### Scenario 2: Interrupt During Poison Generation

```bash
# Experiment running: generating poisoned passages
# Query 35/200 completed, last save at query 30
# [INTERRUPT - Ctrl+C to commute]

# Resume: Same command
uv run python experiments/run_grid.py --datasets all ...

# Output:
# Loading partial poison cache (30/200 queries)
# Continuing from query 31...
# (resumes with 5 queries loss instead of 35)
```

### Scenario 3: Crash After 100/240 Experiments

```bash
# Original run crashes after completing 100 experiments
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all

# Resume shows:
# Loading checkpoint: 100 experiments already completed
# Total experimental conditions: 240
# Already completed: 100
# Remaining to run: 140
# (continues from experiment #101)
```

### Scenario 4: Incremental Execution Across Days

```bash
# Day 1: Run 8 hours, complete 80 experiments
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all
# [Stop for the day]

# Day 2: Resume, complete another 80
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all
# Shows: Already completed: 80, Remaining: 160

# Day 3: Finish remaining 80
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all
# Shows: Already completed: 160, Remaining: 80
```

---

## Benefits

1. **Complete Fault Tolerance**: Survives crashes, power failures, manual interruptions, or commutes
2. **Minimal Work Loss**: Max 5-15 minutes at any interruption point (vs hours before)
3. **Incremental Execution**: Run experiments in multiple sessions across days/weeks
4. **One-Time Costs**: Index building (2-3 hours for NQ) happens only once, cached forever
5. **Cost Efficiency**: Never re-run completed experiments or rebuild completed work
6. **Flexible Scheduling**: Pause and resume around other priorities anytime
7. **Safe Interruption**: Ctrl+C at any time without fear of losing significant work

---

## Maximum Work Loss at Any Interruption Point

| Stage | Before Caching | After Caching |
|-------|---------------|---------------|
| Index building (NQ) | 2-3 hours | 5-10 minutes |
| Poison generation | 30-60 minutes | 10-15 minutes |
| RAG pipeline | Current experiment | Current experiment |
| Future runs (after cache) | 0 (instant load) | 0 (instant load) |

---

## Implementation Details

### Checkpoint System
- **File**: `results/checkpoint.json`
- **Saved when**: After each successful experiment
- **Contains**: Completed experiment IDs, all results, timestamp
- **Experiment ID format**: `{dataset}_{attack}_{defense}_{num_poisoned}`

### Poison Cache
- **Files**: `results/poison_cache/{dataset}_{attack}_{num_poisoned}_{num_queries}.json`
- **Saved when**: Every 10 queries during generation
- **Contains**: Poisoned passages, target answers, doc IDs

### Index Cache
- **Partial files** (during building):
  - `results/index_cache/{dataset}/partial_embeddings.npy`
  - `results/index_cache/{dataset}/partial_doc_ids.npy`
  - `results/index_cache/{dataset}/partial_texts.json`
- **Saved when**: Every 10,000 passages during encoding
- **Final files** (after completion):
  - `results/index_cache/{dataset}/index.faiss`
  - `results/index_cache/{dataset}/doc_ids.npy`
- **Cleanup**: Partial files deleted after successful completion

### Resume Logic
- Automatic in all methods (no manual intervention needed)
- Idempotent: Safe to run the same command multiple times
- Stateless: All state stored on disk, not in memory

---

## Testing

Run the demonstration script to see checkpoint behavior:

```bash
python demo_resume.py
```

Shows:
- Checkpoint creation and saving
- Loading behavior
- Filtering of completed experiments
- Resume simulation

---

## Summary

**You can now interrupt experiments at ANY time and resume with minimal loss:**

✅ **Index building**: Max 10 minutes loss (vs 2-3 hours)  
✅ **Poison generation**: Max 15 minutes loss (vs hours)  
✅ **Experiments**: Restart current experiment only  
✅ **Future runs**: Instant (everything cached)  

**Commands are simple:**
- Start: `uv run python experiments/run_grid.py --datasets all ...`
- Stop: `Ctrl+C` (safe at any time)
- Resume: Same command (automatic)

You're fully protected! 🛡️
