# Resume Functionality

The experiment runner supports automatic checkpointing and resume functionality with three levels of protection:

1. **Experiment-level checkpoints**: After each complete experiment
2. **Incremental poison passage caching**: Saves every 10 queries during generation
3. **FAISS index caching**: Saves built indices per dataset (NEW)

## How It Works

### Level 1: Experiment-level Checkpoints

1. **Automatic Checkpointing**: After each experiment completes successfully, a checkpoint is saved to `results/checkpoint.json`
2. **Resume on Restart**: If you re-run the experiment grid, it automatically detects completed experiments and skips them
3. **Progress Preservation**: All metrics and results are preserved in the checkpoint file

### Level 2: Incremental Poison Passage Caching (NEW)

**The slow part** of each experiment is generating poisoned passages (the batch processing you see). To protect against losing this progress:

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

4. **Maximum Loss**: At most 9 queries worth of work (if interrupted between saves)

**Example**:
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

**This means**: You can safely interrupt at ANY time and lose at most 9 queries worth of work!

### Level 3: FAISS Index Caching (NEW)

**The slowest part** of setup is building the FAISS index (encoding millions of documents). This now gets cached:

1. **Per-dataset Caching**: Each dataset's index is cached separately to `results/index_cache/{dataset_name}/`
   - `nq` index: ~3.5M passages, ~2-3 hours to build
   - `hotpotqa` index: ~400 passages, ~30 seconds
   - `2wikimultihop` index: ~400 passages, ~30 seconds
   - `musique` index: ~4K passages, ~2 minutes

2. **Automatic Reuse**: When running experiments:
   - First run on dataset → builds index + saves to cache
   - Subsequent runs → loads from cache (instant!)
   - Applies to all experiments using that dataset

3. **One-time Cost**: The 2-3 hour index build for NQ happens only once, then is reused forever

**Example**:
```
First run:
  └─ Building index for nq... (2.5 hours)
  └─ ✓ Index cached to results/index_cache/nq/
  
Second run (any experiment using NQ):
  └─ Loading cached index for nq... (2 seconds) ✓
  
All future runs:
  └─ Loading cached index (instant) ✓
```

**This solves**: The 2h 50min you just lost rebuilding the NQ index - it will now be reused!

## Checkpoint File Structure

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

## Usage

Simply run the same command again after an interruption:

```bash
# First run (interrupted after 50 experiments)
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all

# Resume (automatically skips first 50, continues from #51)
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all
```

**Important**: Use the same parameters (datasets, attacks, defenses, num-poisoned) to ensure correct resume behavior.

## Manual Checkpoint Management

### Check Progress

```bash
# View checkpoint status
python -c "
import json
with open('results/checkpoint.json') as f:
    data = json.load(f)
    print(f'Completed: {len(data[\"completed_experiments\"])}')
    print(f'Last updated: {data[\"last_updated\"]}')
"
```

### Reset Checkpoint

If you want to start completely fresh:

```bash
rm results/checkpoint.json
rm results/all_results.json
```

### Clear Poison Cache

If you want to regenerate poisoned passages (e.g., after changing attack code):

```bash
rm -rf results/poison_cache/
```

### Clear Index Cache

If you want to rebuild indices (e.g., after changing chunking parameters):

```bash
rm -rf results/index_cache/
```

### Resume from Partial Checkpoint

The checkpoint is saved after **each successful experiment**, so even if the process crashes mid-run:
- All completed experiments are preserved
- No need to recompute already-finished conditions
- Simply restart with the same command

## Example Scenarios

### Scenario 1: Crash After 100/240 Experiments

```bash
# Original run crashes
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all

# Resume shows:
# Total experimental conditions: 240
# Already completed: 100
# Remaining to run: 140
```

### Scenario 2: Add More Experiments

```bash
# First run: only NQ dataset
uv run python experiments/run_grid.py --datasets nq --attacks all --defenses all
# Completes 60 experiments

# Second run: add more datasets
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all
# Shows: Already completed: 60, Remaining: 180
# Skips the 60 NQ experiments already done
```

## Benefits

1. **Fault Tolerance**: System crashes, power failures, or manual interruptions won't lose progress
2. **Incremental Execution**: Run experiments in batches over multiple sessions
3. **Cost Efficiency**: Don't waste time/compute re-running completed experiments
4. **Flexible Scheduling**: Pause and resume around other priorities

## Technical Details

- Checkpoint file: `{output_dir}/checkpoint.json`
- Saved after: Each successful `run_single_experiment()` call
- Resume check: Automatic in `run_grid()` method
- Experiment ID format: `{dataset}_{attack}_{defense}_{num_poisoned}`
- Idempotent: Safe to run the same command multiple times
