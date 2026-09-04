# SMART: Poisoning-Resilient Memory for Multi-Agent LLMs

## Description

SMART is a trust-aware multimodal Retrieval-Augmented Generation (RAG) pipeline designed to defend shared LLM agent memory against poisoning attacks. The system evaluates and ranks retrieved memory candidates not just by semantic relevance, but by an adaptive trust score fused from multiple evidence dimensions, so that poisoned, low-provenance, or inconsistent content is down-weighted or filtered before it reaches the agent's context.

The pipeline consists of five phases:

1. **Candidate Retrieval** — retrieves candidate memory chunks relevant to a query.
2. **Evidence Extraction** — computes five trust-relevant signals per candidate: `S` (semantic relevance), `P` (provenance), `C` (cross-modal consistency), `U` (confidence), and `R` (safety/risk).
3. **Adaptive Trust Fusion** — combines the five evidence dimensions into a single trust probability using a fitted fusion model.
4. **Trust-Aware Ranking** — reranks candidates using the fused trust score alongside relevance.
5. **Explainable Output** — returns the selected memory along with a trust level and human-readable explanation.

This notebook (`smart-final-code.ipynb`) is a clean, de-duplicated rebuild of the pipeline. Compared to earlier versions, it adds a proper stratified Train/Validation/Test split, and ensures all fitted components (e.g., Mahalanobis covariance estimation, logistic-regression trust correction) are fit only on the training set and applied frozen to validation/test data, eliminating data leakage.

## Dataset Information

Two corpora are used to construct the trusted/untrusted memory pool:

- **Wikipedia** — used as the trusted source corpus.
- **Fakeddit** (`vanshikavmittal/fakeddit-dataset` on Kaggle) — used as the untrusted/noisy source corpus, providing synthetic poisoning examples.

The two corpora are combined and deduplicated to construct the candidate memory pool used for retrieval and trust evaluation. Queries are evaluated against this pool, with ground-truth articles/memory IDs used to compute retrieval and ranking metrics (e.g., Recall@1).

## Code Information

- **Language:** Python 3.12
- **Format:** Jupyter/Colab notebook (`.ipynb`), organized into sequential, labeled phase cells (Phase 0 through Phase 4+) intended to be run top-to-bottom without re-running earlier cells.
- **Core components:**
  - Data loading and corpus construction (`kagglehub` for automatic dataset download)
  - Embedding-based candidate retrieval
  - Evidence extraction across the five trust dimensions
  - Trust fusion model (fit on train split only)
  - Learning-to-rank / trust-aware reranking
  - Evaluation utilities (Recall@1, chunk-level accuracy, ground-truth column detection)

## Usage Instructions

1. Open the notebook in Google Colab, Kaggle, or a local Jupyter environment.
2. Set the `DATA_SOURCE` variable in the configuration cell:
   - `"kagglehub"` — automatically downloads both datasets (default, works in Colab/local/Kaggle).
   - `"kaggle_local"` — use if the datasets are already attached to a Kaggle notebook.
   - `"custom"` — point to your own local copies of the datasets.
3. Run all cells from top to bottom in order. Each phase is self-contained in one clean cell block; no cell needs to be re-run out of sequence.
4. After the pipeline runs, evaluation cells at the end of the notebook report retrieval and ranking metrics (e.g., Recall@1) on the held-out test split.

## Requirements

- Python 3.12+
- `numpy`
- `pandas`
- `scikit-learn`
- `torch`, `torchvision`, `torchaudio`
- `transformers`
- `sentence-transformers`
- `faiss-cpu`
- `xgboost`
- `shap`
- `kagglehub`

Install dependencies with:

```bash
pip install torch torchvision torchaudio transformers sentence-transformers faiss-cpu scikit-learn xgboost shap kagglehub
```

> Note: The notebook includes a setup cell that uninstalls and reinstalls `torch`, `transformers`, and `sentence-transformers` to resolve version conflicts. A session/runtime restart is required immediately after this cell before continuing.

## Methodology

1. **Corpus construction:** Wikipedia (trusted) and Fakeddit (untrusted) data are loaded, cleaned, and deduplicated into a single candidate memory pool.
2. **Data splitting:** A stratified train/validation/test split is applied before any model fitting to prevent leakage.
3. **Evidence extraction:** For each query-candidate pair, five trust-relevant features are computed: semantic relevance (S), provenance (P), cross-modal consistency (C), confidence (U), and safety/risk (R).
4. **Trust fusion:** A fusion model (e.g., Mahalanobis-distance-based covariance estimation and/or logistic regression) is fit exclusively on the training split and then applied, frozen, to validation and test data to produce a trust probability per candidate.
5. **Ranking:** Candidates are reranked using the fused trust score combined with relevance to produce a final ranked list per query.
6. **Evaluation:** Retrieval and ranking quality is assessed using Recall@1 and chunk-level accuracy against ground-truth memory IDs/articles.
7. **Explanation generation:** The top-ranked result for each query is returned with an associated trust level and explanation string.

## Citations

If you use this code or the SMART methodology in your research, please cite the associated manuscript (in preparation / under submission). Citation details will be added upon publication.

## License & Contribution Guidelines

This project is provided for academic and research purposes. Please contact the author before reuse in derivative research or commercial applications. Contributions, issues, and suggestions are welcome via pull request or issue submission.
