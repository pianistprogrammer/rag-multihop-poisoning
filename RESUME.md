# Resume Functionality

The experiment runner supports automatic checkpointing and resume functionality with two levels of protection:

1. **Experiment-level checkpoints**: After each complete experiment
2. **Poison passage caching**: Saves generated poisoned passages to avoid regenerating them

## How It Works

### Level 1: Experiment-level Checkpoints

1. **Automatic Checkpointing**: After each experiment completes successfully, a checkpoint is saved to `results/checkpoint.json`
2. **Resume on Restart**: If you re-run the experiment grid, it automatically detects completed experiments and skips them
3. **Progress Preservation**: All metrics and results are preserved in the checkpoint file

### Level 2: Poison Passage Caching (NEW)

**The slow part** of each experiment is generating poisoned passages (the batch processing you see). To protect against losing this progress:

1. **Automatic Caching**: After generating poisoned passages for an experiment, they're cached to `results/poison_cache/{dataset}_{attack}_{num_poisoned}_{num_queries}.json`
2. **Reuse on Restart**: If the experiment is interrupted during the RAG pipeline phase, restarting will:
   - Load cached poisoned passages (instant)
   - Skip the slow generation phase
   - Continue with the RAG evaluation
3. **Per-configuration Cache**: Each unique combination of (dataset, attack, num_poisoned, num_queries) gets its own cache file

**This means**: Even if you stop the process at 29% batches, the next time you run that experiment configuration, it will use the cache and skip directly to where it was interrupted!

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
