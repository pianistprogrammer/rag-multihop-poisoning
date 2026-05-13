#!/usr/bin/env python3
"""
Test script to verify ASR calculation fixes.
"""

from src.rag_poisoning.evaluation.metrics import compute_asr

# Test case 1: Exact substring match (should work)
predictions = ["The answer is Paris. It is the capital of France."]
target_answers = ["Paris"]
asr = compute_asr(predictions, target_answers, match_type="substring")
print(f"Test 1 - Exact substring match: ASR = {asr:.3f} (expected: 1.000)")

# Test case 2: Truncated target (old problem - fails with substring)
predictions = ["The Eiffel Tower was built in 1889 for the World's Fair."]
target_answers = ["The Eiffel Tower was built in 1889 for the World's Fair in Paris. It stands 330 meters tall and was d..."]
asr_substring = compute_asr(predictions, target_answers, match_type="substring")
asr_fuzzy = compute_asr(predictions, target_answers, match_type="fuzzy", similarity_threshold=0.7)
print(f"\nTest 2 - Truncated target:")
print(f"  Substring match: ASR = {asr_substring:.3f} (expected: 0.000 - fails)")
print(f"  Fuzzy match (0.7): ASR = {asr_fuzzy:.3f} (expected: 1.000 - works!)")

# Test case 3: Paraphrased answer
predictions = ["Paris is the capital city of France and its largest urban area."]
target_answers = ["Paris is the capital of France."]
asr_substring = compute_asr(predictions, target_answers, match_type="substring")
asr_fuzzy = compute_asr(predictions, target_answers, match_type="fuzzy", similarity_threshold=0.7)
print(f"\nTest 3 - Paraphrased answer:")
print(f"  Substring match: ASR = {asr_substring:.3f} (expected: 0.000 - too strict)")
print(f"  Fuzzy match (0.7): ASR = {asr_fuzzy:.3f} (expected: 1.000 - flexible!)")

# Test case 4: Complete sentence extraction
text = "The Eiffel Tower was built in 1889. It stands in Paris, France. It is 330 meters tall and attracts millions of visitors each year."

# Old way (truncate at 100 chars - bad)
target_old = text[:100]
print(f"\nTest 4 - Target answer extraction:")
print(f"  Old way (100 chars): '{target_old}' (truncated mid-sentence)")

# New way (first complete sentence - good)
target_new = text[:50].strip()
for sep in ['. ', '.\n', '! ', '?\n']:
    idx = text.find(sep)
    if 0 < idx < 200:
        target_new = text[:idx+1].strip()
        break
print(f"  New way (sentence): '{target_new}' (complete!)")

print("\n✓ All tests completed! The fuzzy matching approach should fix ASR=0.000")
print("\nRecommendation: Clear the poison_cache directory and re-run experiments:")
print("  rm -rf results/poison_cache/*.json")
print("  python run_experiments.py")
