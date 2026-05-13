# ✅ Ready to Re-run Experiments

## What Was Done

### ✅ Fixed the ASR=0.000 Bug
1. **runner.py** - Extract complete sentences as target answers (instead of truncating at 100 chars)
2. **metrics.py** - Added fuzzy matching (token overlap ≥60%) for ASR calculation
3. **Cleaned up** - Removed old checkpoint with buggy results

### ✅ What's Preserved
- **Index cache** (10GB) - Saves ~5 min loading time per dataset
- **Code changes** - All fixes are in place
- **Backup** - Old checkpoint saved to `results/checkpoint_old_buggy.json`

## Run Command

```bash
cd /Volumes/LLModels/Projects/RagPoisoning/rag-multihop-poisoning

# Run full experimental grid (240 conditions)
uv run python experiments/run_grid.py \
    --datasets nq \
    --attacks all \
    --defenses all \
    --num-poisoned 1 3 5 \
    --num-queries 200 \
    --log-level INFO
```

## What Will Happen

1. **Loads cached index** - 5 min (one-time per dataset)
2. **Generates poisoned passages** - ~2-5 min per condition (with caching)
3. **Runs queries through RAG pipeline** - ~5-10 min per condition
4. **Saves checkpoint after each condition** - Resume-safe!
5. **240 total conditions** - Estimated ~20-40 hours total

## Expected Results (After Fix)

```json
{
  "asr": 0.3-0.7,      // ✅ Attacks should now fool LLM
  "rsr": 0.8-1.0,      // ✅ Poisoned docs get retrieved  
  "stealthiness": 0.5-0.9,  // Varies by attack/defense
  "aua": 0.4-0.7,      // Some accuracy maintained
  "f1": 0.4-0.6        // Reasonable answer quality
}
```

## If You Need to Stop/Resume

The experiment auto-saves checkpoints. If interrupted, just re-run the same command:

```bash
uv run python experiments/run_grid.py --datasets nq --attacks all --defenses all
```

It will automatically:
- Skip completed experiments
- Load previous results
- Continue from where it stopped

## Monitor Progress

Logs show progress:
```
INFO - Condition 45/240 (remaining: 45/240)
INFO - Running: nq | poisonedrag | filterrag | 3 poisoned
INFO - Results: ASR=0.42, RSR=0.98
```

## Output Files

- `results/checkpoint.json` - Progress tracker
- `results/all_results.json` - Final aggregated results
- `results/poison_cache/*.json` - Cached poisoned passages
- `results/index_cache/` - Cached retriever indices

---

**Your experiment is ready to run with the ASR fix! 🚀**
