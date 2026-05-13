# Proposal 1 — RAG Poisoning Under Multi-Hop and Chain-of-Evidence Attack Conditions

**Document type:** Research PRD  
**Status:** Draft v1.0  
**Author:** Research proposal document  
**Target venues:** EMNLP, ACL (Security/Safety Track), USENIX Security  

---

## 1. Executive Summary

Existing RAG poisoning research has been conducted almost exclusively on single-hop question-answering benchmarks. The real world increasingly uses multi-hop RAG — where answering a query requires chaining evidence across multiple retrieved documents — yet no published study has systematically evaluated whether poisoning attacks are more or less effective in this setting, nor whether standard defenses hold up when the attack is wrapped in a plausible reasoning chain. This proposal fills that gap by designing, implementing, and evaluating chain-of-evidence poisoning attacks against multi-hop RAG pipelines using open-source 7B-class models.

---

## 2. Problem Statement

### 2.1 Background

Retrieval-Augmented Generation (RAG) systems enhance LLMs by grounding generation in externally retrieved documents. Knowledge poisoning attacks (e.g. PoisonedRAG, CorruptRAG) inject adversarially crafted passages into the knowledge base to induce the LLM to produce attacker-chosen outputs for targeted queries.

All published poisoning attacks and defenses have been evaluated on flat, single-hop QA corpora — primarily Natural Questions (NQ), HotpotQA (treated as single-step), and MS-MARCO. In these settings, a poisoned passage needs to be semantically close to a single query to be retrieved and consumed by the generator.

### 2.2 The Multi-Hop Gap

Multi-hop RAG systems retrieve *multiple* supporting documents per query and ask the LLM to synthesise an answer across them. This introduces qualitatively different attack dynamics:

- An attacker may need only one poisoned document in a chain to corrupt the final answer.
- Chain-of-evidence attacks — where misinformation is wrapped in a step-by-step reasoning narrative aligned with the model's instruction-following style — may be significantly harder for existing detectors to catch.
- Existing defenses (FilterRAG, RAGDefender, RAGPart) have been designed and tested only against flat single-hop poisoning; their efficacy against multi-hop variants is unknown.

### 2.3 Research Gap Statement

> There is no published systematic evaluation of RAG poisoning attacks against multi-hop reasoning pipelines using open-source small models, nor any characterisation of how chain-of-evidence narrative wrapping affects attack success rate and stealthiness.

This gap has been acknowledged explicitly in the Semantic Chameleon (2026) and RAG Security Bench (2025) papers but not addressed.

---

## 3. Research Objectives

| ID | Objective |
|----|-----------|
| O1 | Measure and compare ASR of standard poisoning attacks (PoisonedRAG, CorruptRAG) on single-hop vs. multi-hop RAG pipelines under identical conditions |
| O2 | Design and evaluate a chain-of-evidence (CoE) attack variant that wraps poisoned content in a structured reasoning narrative to evade detection |
| O3 | Evaluate whether existing retrieval-stage and generation-stage defenses transfer effectively to multi-hop and CoE attack conditions |
| O4 | Characterise the relationship between reasoning chain depth (1-hop, 2-hop, 4-hop) and attack success rate and stealthiness |

---

## 4. Research Questions

- **RQ1:** Does attack success rate differ systematically between single-hop and multi-hop RAG settings when the generator is a 7B open-source model?
- **RQ2:** Does chain-of-evidence wrapping of poisoned content elevate ASR beyond flat injection, and by how much?
- **RQ3:** Do standard defenses (FilterRAG, RAGDefender, RAGPart, RAGMask) maintain their published performance under multi-hop and CoE attack conditions?
- **RQ4:** Is there a hop-depth threshold above which poisoning becomes significantly easier or harder?

---

## 5. Datasets

### 5.1 Primary Evaluation Corpora

| Dataset | Reasoning type | Corpus size | Hops | Source |
|---------|---------------|-------------|------|--------|
| **HotpotQA** | Two-document multi-hop | 5.23M passages | 2 | HuggingFace `hotpot_qa` |
| **2WikiMultiHopQA** | Two-document with explicit supporting facts | 430K passages | 2 | HuggingFace `voidful/2wikimultihop` |
| **MuSiQue** | Four-hop compositional | 139K passages | 4 | HuggingFace `musique` |
| **Natural Questions (NQ)** | Single-hop factoid | 2.68M passages (BEIR format) | 1 | `beir` library |

### 5.2 Dataset Preparation

- All corpora converted to BEIR format (standardised `corpus.jsonl` / `queries.jsonl` / `qrels.tsv` schema).
- For multi-hop datasets, supporting fact annotations are used to identify the exact documents that must be retrieved for a correct answer — these define the attack target positions.
- Evaluation split: 200 targeted queries per dataset (sampled from the test splits), matching the experimental scale of PoisonedRAG.

### 5.3 Why These Datasets

NQ is the established single-hop baseline enabling direct comparison to all prior work. HotpotQA is the most widely used multi-hop benchmark. 2WikiMultiHopQA provides explicit supporting fact labels that enable precise control of which chain position is poisoned. MuSiQue extends to four-hop chains, allowing study of reasoning depth effects.

---

## 6. Proposed Attack: Chain-of-Evidence (CoE) Poisoning

### 6.1 Standard Attack Baselines

Replicate the following as baselines:

- **PoisonedRAG (black-box):** LM-targeted passage generation; injects `adv_per_query=5` passages. Uses original codebase from `github.com/sleeepeer/PoisonedRAG`.
- **CorruptRAG:** Single-document injection; higher stealthiness constraint.
- **PIDP-Attack:** Combined prompt injection + database poisoning.

### 6.2 Novel Attack: Chain-of-Evidence (CoE) Variant

**Core idea:** Standard attacks inject a passage that directly states the target answer. CoE attacks instead generate a passage that *reasons toward* the target answer using a structured multi-step narrative — mimicking the style of legitimate multi-hop reasoning chains found in the corpus.

**CoE passage structure:**

```
[Premise 1]: [Factual-sounding claim about entity A]
[Premise 2]: [Factual-sounding claim linking A to B]
[Inference]: [Therefore, the answer to [TARGET QUESTION] is [TARGET ANSWER]]
[Authority signal]: [Invented citation or authoritative reference]
```

**Generation process:**

1. Use a local 7B model (Mistral-7B-Instruct) to generate the CoE passage given: target question, target answer, a few real corpus passages as style exemplars.
2. Apply embedding similarity maximisation (same objective as PoisonedRAG's LM-targeted method) to ensure retrieval.
3. Optionally apply a perplexity constraint so the passage is not flagged by perplexity-based detectors.

**Rationale:** Multi-hop LLMs are trained to trust and follow reasoning chains. A poisoned passage that mimics this format is more likely to be trusted by the generator even when it contradicts parametric knowledge, and harder for detection methods relying on semantic anomaly to catch.

---

## 7. Experimental Design

### 7.1 RAG Pipeline

```
Query → Dense Retriever (Contriever / BGE-base) → FAISS Index 
      → Top-k Passages (k=5) → Generator (7B LLM) → Answer
```

**Retriever:** `facebook/contriever-msmarco` (dense); BM25 (sparse, via `pyserini`) for hybrid conditions.  
**Generator:** Mistral-7B-Instruct-v0.3 via Ollama.  
**Vector store:** FAISS `IndexFlatIP` (exact inner product search).  
**Chunk size:** 100 words, matching PoisonedRAG's original setup.

### 7.2 Experiment Grid

| Variable | Values |
|----------|--------|
| Dataset | NQ, HotpotQA, 2WikiMultiHopQA, MuSiQue |
| Attack | PoisonedRAG-BB, CorruptRAG, PIDP, CoE (proposed) |
| Defense | No defense, FilterRAG, RAGDefender, RAGPart |
| Poisoned docs per query | 1, 3, 5 |

Total conditions: 4 datasets × 4 attacks × 4 defense conditions × 3 injection levels = 192 conditions.  
Per condition: 200 targeted queries → approximately 38,400 individual evaluations.

### 7.3 Chain Position Experiment (Multi-Hop Only)

For HotpotQA and 2WikiMultiHopQA, additionally vary *which hop* in the chain is poisoned:

- **Position A:** Poison the first supporting document.
- **Position B:** Poison the second supporting document.
- **Position AB:** Poison both.

This isolates whether early or late chain position matters for attack success.

---

## 8. Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **Attack Success Rate (ASR)** | Fraction of target queries where LLM output contains the target answer (substring match) |
| **Retrieval Success Rate (RSR)** | Fraction of queries where at least one poisoned passage appears in top-k retrieved set |
| **Stealthiness Score** | 1 − (fraction of poisoned passages flagged by detector) |
| **Accuracy Under Attack (AUA)** | Fraction of non-targeted queries answered correctly (measures collateral damage) |
| **Defense Effectiveness (DE)** | Reduction in ASR under each defense condition vs. no-defense baseline |
| **F1 Score** | Exact match F1 for CoE vs. baseline attack, following PoisonedRAG protocol |

---

## 9. Hypotheses

- **H1:** Multi-hop RAG will exhibit *higher* ASR than single-hop for the same number of injected passages, because retrieving multiple documents per query increases the probability of including a poisoned one.
- **H2:** CoE attacks will achieve higher ASR than flat injection attacks on multi-hop datasets, because multi-hop generators are conditioned to follow reasoning chains.
- **H3:** Existing defenses will show significantly degraded effectiveness on CoE attacks compared to their published performance on flat attacks.
- **H4:** Poisoning the first hop of a two-hop chain will be sufficient to corrupt the final answer in the majority of cases.

---

## 10. Technical Implementation Plan

### 10.1 Environment

```
Hardware:   Apple M4, 48GB unified memory
OS:         macOS (Apple Silicon)
Python:     3.11
LLM server: Ollama (local inference)
Model:      Mistral-7B-Instruct-v0.3 (Q4_K_M quantisation, ~4.1GB)
Retriever:  sentence-transformers + FAISS (CPU, MPS-accelerated)
```

### 10.2 Dependencies

```
pip install beir faiss-cpu sentence-transformers datasets
pip install pyserini  # BM25
ollama pull mistral:7b-instruct
```

### 10.3 Repository Structure

```
rag-multihop-poisoning/
├── data/
│   ├── download_datasets.py       # BEIR format download
│   └── prepare_multihop.py        # Supporting fact extraction
├── attacks/
│   ├── poisonedrag_bb.py          # Baseline black-box attack
│   ├── corruptrag.py              # Single-injection baseline
│   ├── pidp_attack.py             # PIDP baseline
│   └── coe_attack.py              # Proposed chain-of-evidence attack
├── pipeline/
│   ├── retriever.py               # FAISS + Contriever wrapper
│   ├── generator.py               # Ollama inference wrapper
│   └── rag_pipeline.py            # Full pipeline assembly
├── defenses/
│   ├── filterrag.py
│   ├── ragdefender.py
│   └── ragpart.py
├── evaluation/
│   ├── metrics.py                 # ASR, RSR, stealthiness, AUA
│   └── run_experiment.py          # Grid experiment runner
├── results/
│   └── analysis.ipynb
└── README.md
```

### 10.4 Timeline

| Week | Milestone |
|------|-----------|
| 1–2 | Dataset download and preprocessing; BEIR format validation |
| 3–4 | Baseline attack reimplementation and validation against published NQ numbers |
| 5–6 | CoE attack design and generation pipeline |
| 7–8 | Full experiment grid execution |
| 9 | Defense evaluation |
| 10 | Analysis, ablations, hypothesis testing |
| 11–12 | Paper writing |

---

## 11. Expected Contributions

1. **Empirical finding:** First characterisation of how reasoning chain depth affects RAG poisoning ASR and stealthiness using open-source models.
2. **Novel attack:** Chain-of-evidence poisoning method with open-source implementation.
3. **Defense evaluation:** Evidence for or against transfer of existing defenses to multi-hop conditions.
4. **Benchmark:** Standardised multi-hop RAG poisoning evaluation protocol compatible with the BEIR ecosystem.

---

## 12. Limitations and Mitigations

| Limitation | Mitigation |
|------------|-----------|
| Single generator model | Run ablation with Qwen2.5-7B as secondary generator |
| Black-box attack focus | Include white-box HotFlip as supplementary baseline |
| English-only corpora | Scope limitation stated explicitly; future work note on multilingual |
| Computational budget | Grid can be parallelised across datasets; each condition takes ~2h on M4 |

---

## 13. Related Work Summary

| Paper | Venue | Relevance |
|-------|-------|-----------|
| PoisonedRAG (Zou et al.) | USENIX Security 2025 | Primary attack baseline |
| CorruptRAG (2025) | arXiv | Single-injection baseline |
| AuthChain (2025) | arXiv | Multi-hop single-injection attack; direct prior work |
| RAG Security Bench (2025) | arXiv | Benchmark; 13 attacks, 7 defenses |
| Semantic Chameleon (2026) | arXiv | Corpus-dependency finding |
| RAGPart & RAGMask (2025) | arXiv | Retrieval-stage defenses |
| FilterRAG / RAGDefender (2025) | arXiv | Generation-stage defenses |

---

## 14. Ethical Considerations

All experiments are conducted on publicly available datasets in a closed, offline environment. No attacks are mounted against live or production systems. The attack code will be released with responsible disclosure documentation. Findings will be reported in a manner that prioritises defensive insight.

---

*End of Proposal 1 PRD*
