# Note–IdeaUnit Coverage Pipeline (Rule / BERT-LR / LLM / LLM-RAG)

This project runs a full pipeline for **zero-shot idea-unit coverage classification**:
- Merge `Notes.csv` with `train.csv` and `test.csv`
- Train/evaluate:
  - Rule-based lexical baseline
  - BERT embeddings + Logistic Regression (plus ensemble)
  - LLM (plain ICL) and LLM-RAG (retrieval-augmented ICL)
- Export metrics + prediction CSVs

---

## Dependencies

```bash
pip install pandas scikit-learn sentence-transformers faiss-cpu tqdm lmdeploy
```

> If you do not want FAISS, you can skip `faiss-cpu`. RAG will fall back to a numpy similarity search.

---

## Files

**Inputs**
- `Notes.csv`: raw student notes with `Segment1_Notes ... Segment4_Notes`
- `train.csv`: labeled pairs `(Experiment, Topic, ID, Segment, IdeaUnit, label)`
- `test.csv`: test pairs (may include partial labels)

**Outputs**
- `train_merged.csv`, `test_merged.csv`: merged datasets with `NoteText`
- `outputs/metrics_overall.csv`
- `outputs/metrics_by_topic.csv`
- `outputs/metrics_report.txt`
- `outputs/submission_bert_ensemble.csv` (if BERT is run)

---

## Hugging Face Token (Optional)

If your models require authentication:

```bash
export HF_TOKEN="YOUR_TOKEN"
export TOKENIZERS_PARALLELISM=false
```

⚠️ Do **not** hardcode tokens in code.

---

## Usage

### Run everything (Rule + BERT + LLM)
```bash
python run_pipeline.py --run-rule --run-bert --run-llm
```

### Run only Rule + BERT (fast)
```bash
python run_pipeline.py --run-rule --run-bert
```

### Run only BERT (and save ensemble predictions)
```bash
python run_pipeline.py --run-bert
```

### Run only LLM (slow)
```bash
python run_pipeline.py --run-llm
```

---

## Common CLI Flags

### Inputs / outputs
- `--notes Notes.csv --train train.csv --test test.csv`
- `--out-dir outputs`
- `--train-merged train_merged.csv --test-merged test_merged.csv`
- `--skip-merge` (reuse existing merged CSVs)

### Train/valid split
- `--seed 42`
- `--train-frac 0.8`

---

## BERT (Embeddings + Logistic Regression)

Default models:
- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`

Useful flags:
```bash
--bert-models sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-mpnet-base-v2 \
--bert-threshold 0.5 \
--bert-embed-bs 32 \
--bert-C 1.0 \
--bert-max-iter 200
```

Notes:
- The classifier uses **log loss (binary cross-entropy)** internally via scikit-learn Logistic Regression.
- High accuracy + low F1 can happen when the model predicts mostly negatives (class imbalance + threshold effects).

---

## LLM (lmdeploy)

Model + inference settings:
```bash
--llm-model meta-llama/Meta-Llama-3-8B-Instruct \
--llm-max-num-seqs 4 \
--llm-max-model-len 1024 \
--llm-extra-context 1024 \
--llm-max-new-tokens 32 \
--temperature 0.7 \
--top-p 0.95 \
--min-p 0.01
```

Notes:
- LLM inference is usually the slowest component.
- Use `--run-llm` only when needed.

---

## RAG Settings

```bash
--rag-encoder sentence-transformers/all-MiniLM-L6-v2 \
--rag-top-k 4 \
--rag-fewshot-k 4 \
--plain-fewshot-k 6
```

RAG uses **pair-level retrieval**:
- Query: `IdeaUnit [SEP] NoteText`
- Retrieve top-k similar labeled `(IdeaUnit, NoteText, label)` examples.

---

## Example: Full Run with Custom Paths

```bash
python run_pipeline.py \
  --notes /path/to/Notes.csv \
  --train /path/to/train.csv \
  --test /path/to/test.csv \
  --out-dir outputs \
  --run-rule --run-bert --run-llm
```
