# 🔄 Restart Instructions - Using Full NQ Dataset

## What Changed:
1. ✅ Fixed ASR metric (fuzzy matching instead of substring)
2. ✅ Fixed AUA metric (fuzzy matching instead of substring)  
3. ✅ Fixed target answer extraction (complete sentences vs truncation)
4. ✅ Changed from `nq_processed` (5 queries) to `nq` (3,452 queries)

## Why Restart:
Your current 20 experiments used only 5 queries from `nq_processed`.
This is not statistically valid. Need to re-run on full dataset.

## What's Preserved:
✅ **Index cache** (21GB) - No need to rebuild embeddings!
✅ **Code fixes** - ASR/AUA fuzzy matching applied
✅ **Poison cache** - Will regenerate with new queries (they'll be different)

## Commands:

```bash
cd /Volumes/LLModels/Projects/RagPoisoning/rag-multihop-poisoning

# Backup old results
cp results/checkpoint.json results/checkpoint_5queries_INVALID.json

# Clear checkpoint for fresh start
rm results/checkpoint.json

# Optional: Clear poison cache (will use different queries now)
rm -rf results/poison_cache/*.json

# Start full experiment with 200 queries
uv run python experiments/run_grid.py \
    --datasets nq \
    --attacks all \
    --defenses all \
    --num-queries 200 \
    --log-level INFO
```

## Expected:
- Total: 240 experiments
- Time: ~20-40 hours (depends on Ollama speed)
- Index loading: 5 min (cached)
- Results: Valid, statistically sound

## Progress Tracking:
Watch for:
```
INFO - Loaded checkpoint: 0 experiments already completed
INFO - Total experimental conditions: 240
INFO - Condition 1/240 (remaining: 1/240)
```

You should see queries being sampled from 3,452 instead of 5!
