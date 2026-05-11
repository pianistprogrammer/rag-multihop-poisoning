# Resume Functionality Implementation Summary

## Changes Made

### 1. Updated `src/rag_poisoning/evaluation/runner.py`

**Added checkpoint management:**
- `checkpoint_file`: Path to `results/checkpoint.json`
- `completed_experiments`: Set of completed experiment IDs
- `_load_checkpoint()`: Loads previous checkpoint on initialization
- `_save_checkpoint()`: Saves checkpoint after each successful experiment

**Modified `run_grid()` method:**
- Filters out completed experiments before running
- Shows progress: "Already completed: X, Remaining: Y"
- Continues from where it left off automatically

**Modified `run_single_experiment()` method:**
- Calls `_save_checkpoint()` after successful completion
- Ensures progress is saved immediately

### 2. Updated `experiments/run_grid.py`

**Added `--resume` flag:**
- Documents that resume is automatic (always enabled)
- No need to explicitly pass the flag

### 3. Created Documentation

**`RESUME.md`:**
- Comprehensive guide on resume functionality
- Usage examples and scenarios
- Manual checkpoint management commands

**`demo_resume.py`:**
- Demonstrates checkpoint save/load behavior
- Shows filtering of completed experiments

## How to Use

### Normal Usage (Automatic Resume)

```bash
# First run (interrupted after 100 experiments)
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all

# Second run (automatically resumes from experiment #101)
uv run python experiments/run_grid.py --datasets all --attacks all --defenses all
```

### Check Progress

```bash
# View checkpoint
cat results/checkpoint.json | jq '.completed_experiments | length'
```

### Reset and Start Fresh

```bash
rm -rf results/
```

## Benefits

1. **Fault Tolerance**: Survives crashes, power failures, interruptions
2. **Incremental Execution**: Run in multiple sessions
3. **No Wasted Compute**: Never re-runs completed experiments
4. **Safe Interruption**: Ctrl+C at any time, resume later

## Implementation Details

- **Checkpoint format**: JSON with completed IDs, results, and timestamp
- **Saved when**: After each successful experiment (not batch)
- **Experiment ID**: `{dataset}_{attack}_{defense}_{num_poisoned}`
- **Resume logic**: Automatic filtering in `run_grid()`
- **Idempotent**: Safe to run same command multiple times

## Example Output

```
Loading checkpoint: 50 experiments already completed
Restored 50 previous results

Total experimental conditions: 240
Already completed: 50
Remaining to run: 190

Condition 51/240 (remaining: 1/190)
================================================================================
Running: hotpotqa | coe | filterrag | 3 poisoned
...
```

## Testing

Run the demonstration:
```bash
python demo_resume.py
```

Shows:
- Checkpoint creation
- Loading behavior  
- Filtering of completed experiments
- Resume simulation
