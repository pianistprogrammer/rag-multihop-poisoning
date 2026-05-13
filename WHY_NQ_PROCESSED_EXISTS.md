# 🎯 Final Answer: Why `nq_processed` Exists

## The Design Intent:

`nq_processed/` was created by the preprocessing script to:
1. **Download** the full BEIR NQ dataset
2. **Sample** a manageable subset of queries (default: 200)
3. **Save** in a simplified flat format for experiments

## What Went Wrong:

Someone ran the preprocessing with `--num-queries 5` (probably for quick testing):
```bash
python experiments/prepare_data.py --datasets nq --num-queries 5
```

This created `nq_processed/` with:
- ✅ Full corpus (2.68M docs)
- ❌ Only 5 queries (instead of 200 or full 3,452)
- ❌ Only 5 qrel mappings

## The Two Datasets:

| Dataset | Queries | Purpose | Format |
|---------|---------|---------|--------|
| `nq/` | 3,452 | **Full BEIR** | Raw BEIR (`qrels/test.tsv`) |
| `nq_processed/` | 5 | Debug subset | Processed flat (`qrels.tsv`) |

## The Fix:

I've made **two changes**:

### 1. ✅ Fixed `BEIRDataset.load()` to handle both formats
Now it checks for both:
- `qrels.tsv` (processed format)
- `qrels/test.tsv` (BEIR raw format)

### 2. ✅ Changed `run_grid.py` to use `nq/` instead of `nq_processed/`
```python
# OLD (line 139):
data_path = Path(args.data_dir) / f"{dataset_name}_processed"  # Uses nq_processed (5 queries)

# NEW:
data_path = Path(args.data_dir) / dataset_name  # Uses nq (3,452 queries)
```

## Summary:

**`nq_processed` was created as a preprocessing cache, but was accidentally generated with only 5 queries for debugging. Now we use the full `nq/` dataset with 3,452 queries.**

## Your Next Steps:

```bash
cd /Volumes/LLModels/Projects/RagPoisoning/rag-multihop-poisoning

# Backup old results (5-query experiments)
mv results/checkpoint.json results/checkpoint_5queries_debug.json

# Clear for fresh start
rm results/checkpoint.json
rm -rf results/poison_cache/*.json

# Run full experiment with 3,452 queries available (sampling 200)
uv run python experiments/run_grid.py \
    --datasets nq \
    --attacks all \
    --defenses all \
    --num-queries 200 \
    --log-level INFO
```

Now your experiments will properly sample 200 queries from the full 3,452 available! 🎉
