# Centralized Dataset Organization

## Overview

All RAG poisoning datasets are stored in a centralized location to enable reuse across multiple projects:

```
/Volumes/LLModels/Datasets/RAG/
```

## Benefits

1. **Reusability**: Download datasets once, use across multiple research projects
2. **Space Efficiency**: Avoid duplicate dataset copies
3. **Version Control**: Single source of truth for processed datasets
4. **Easy Sharing**: Consistent path for collaboration

## Current Structure

```
/Volumes/LLModels/Datasets/RAG/
├── nq/                      # Raw Natural Questions from BEIR
├── nq.zip                   # Downloaded archive
├── nq_processed/            # Processed BEIR format
│   ├── corpus.jsonl         # 2.68M documents
│   ├── queries.jsonl        # 5 queries (test sample)
│   └── qrels.tsv            # Query-document relevance
├── hotpotqa_processed/      # (To be downloaded)
├── 2wikimultihop_processed/ # (To be downloaded)
└── musique_processed/       # (To be downloaded)
```

## Dataset Details

| Dataset | Type | Docs | Queries | Size | Status |
|---------|------|------|---------|------|--------|
| **Natural Questions** | Single-hop | 2.68M | 5 (sample) | ~500MB | ✅ Downloaded |
| **HotpotQA** | 2-hop | 5.23M | TBD | ~1.2GB | ⏳ Pending |
| **2WikiMultiHopQA** | 2-hop | 430K | TBD | ~200MB | ⏳ Pending |
| **MuSiQue** | 4-hop | 139K | TBD | ~150MB | ⏳ Pending |

**Total estimated space**: ~2-3GB after processing

## Usage in Projects

### This Project (rag-multihop-poisoning)

All scripts default to the centralized location:

```bash
# Prepare datasets (saves to /Volumes/LLModels/Datasets/RAG/)
uv run python experiments/prepare_data.py --datasets all

# Run experiments (loads from /Volumes/LLModels/Datasets/RAG/)
uv run python experiments/run_grid.py --datasets all
```

### Override Path (if needed)

```bash
# Use custom location
uv run python experiments/run_grid.py \
    --data-dir /path/to/custom/location \
    --datasets nq
```

### From Python Code

```python
from rag_poisoning.data.datasets import BEIRDataset

# Load from centralized location
dataset = BEIRDataset.load("/Volumes/LLModels/Datasets/RAG/nq_processed")
print(f"Loaded: {len(dataset.corpus)} docs, {len(dataset.queries)} queries")
```

## Adding New Datasets

To add a new RAG dataset to the centralized repository:

1. **Download and process** into BEIR format:
   ```bash
   uv run python experiments/prepare_data.py \
       --datasets <dataset_name> \
       --output-dir /Volumes/LLModels/Datasets/RAG
   ```

2. **Verify structure**:
   ```
   /Volumes/LLModels/Datasets/RAG/<dataset_name>_processed/
   ├── corpus.jsonl
   ├── queries.jsonl
   └── qrels.tsv
   ```

3. **Document** in this file with:
   - Dataset name and type
   - Document count
   - Query count
   - Size estimate

## Sharing Across Projects

To use these datasets in another project:

1. **Reference the centralized path**:
   ```python
   DATA_DIR = "/Volumes/LLModels/Datasets/RAG"
   ```

2. **Use the same BEIR loader**:
   ```python
   from rag_poisoning.data.datasets import BEIRDataset
   dataset = BEIRDataset.load(f"{DATA_DIR}/nq_processed")
   ```

3. **Or implement your own loader** following BEIR format:
   - `corpus.jsonl`: `{"_id": str, "title": str, "text": str}`
   - `queries.jsonl`: `{"_id": str, "text": str}`
   - `qrels.tsv`: `query-id\tcorpus-id\tscore`

## Backup and Versioning

**Important**: The centralized dataset location is **not** git-tracked.

**Backup strategy**:
- Raw datasets (`.zip` files): Keep for reproducibility
- Processed datasets: Can be regenerated from raw
- Consider external backup for large datasets

**Version tracking**:
- Document any preprocessing changes
- Note dataset version/date in this file
- Use semantic versioning for major changes

## Access from Remote Systems

If working on a different machine:

1. **Option A: Sync datasets**
   ```bash
   rsync -avz /Volumes/LLModels/Datasets/RAG/ remote:/path/to/datasets/RAG/
   ```

2. **Option B: Mount network drive**
   ```bash
   # Configure path in configs/default.yaml
   paths:
     data_dir: "/mnt/shared/Datasets/RAG"
   ```

3. **Option C: Re-download**
   ```bash
   uv run python experiments/prepare_data.py --datasets all
   ```

## Migration Notes

**2026-05-11**: Migrated datasets from project-local `./data/` to centralized `/Volumes/LLModels/Datasets/RAG/`
- NQ dataset moved successfully
- All scripts updated to use centralized path
- `.gitignore` updated to exclude only `results/`
