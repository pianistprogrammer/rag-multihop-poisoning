#!/bin/bash
# Verification script for RAG poisoning project setup

echo "=========================================="
echo "RAG Poisoning Project Setup Verification"
echo "=========================================="
echo ""

# Check centralized dataset directory
echo "1. Checking centralized dataset directory..."
DATASET_DIR="/Volumes/LLModels/Datasets/RAG"
if [ -d "$DATASET_DIR" ]; then
    echo "   ✓ Dataset directory exists: $DATASET_DIR"
    echo "   Contents:"
    ls -lh "$DATASET_DIR" | grep -v "^total" | awk '{print "     -", $9, "(" $5 ")"}'
else
    echo "   ✗ Dataset directory not found: $DATASET_DIR"
fi
echo ""

# Check NQ dataset
echo "2. Checking NQ dataset..."
NQ_DIR="$DATASET_DIR/nq_processed"
if [ -d "$NQ_DIR" ]; then
    echo "   ✓ NQ dataset found"
    if [ -f "$NQ_DIR/corpus.jsonl" ]; then
        CORPUS_LINES=$(wc -l < "$NQ_DIR/corpus.jsonl")
        echo "     - corpus.jsonl: $CORPUS_LINES documents"
    fi
    if [ -f "$NQ_DIR/queries.jsonl" ]; then
        QUERIES_LINES=$(wc -l < "$NQ_DIR/queries.jsonl")
        echo "     - queries.jsonl: $QUERIES_LINES queries"
    fi
else
    echo "   ✗ NQ dataset not found"
fi
echo ""

# Check Python environment
echo "3. Checking Python environment..."
if command -v uv &> /dev/null; then
    echo "   ✓ uv package manager installed"
else
    echo "   ✗ uv not found - install from https://github.com/astral-sh/uv"
fi
echo ""

# Check Ollama
echo "4. Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "   ✓ Ollama installed"
    if ollama list | grep -q "mistral:7b-instruct"; then
        echo "   ✓ mistral:7b-instruct model available"
    else
        echo "   ⚠ mistral:7b-instruct not found - run: ollama pull mistral:7b-instruct"
    fi
else
    echo "   ✗ Ollama not found - install from https://ollama.ai"
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Status"
echo "=========================================="
echo "✓ = Ready"
echo "⚠ = Warning (optional)"
echo "✗ = Action required"
echo ""
echo "Next steps:"
echo "1. Download remaining datasets: uv run python experiments/prepare_data.py --datasets all"
echo "2. Run pilot experiment: uv run python experiments/test_single.py"
echo "3. Run full grid: uv run python experiments/run_grid.py"
