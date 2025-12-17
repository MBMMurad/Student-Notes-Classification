import os
import re
import gc
import random
import argparse
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Optional: FAISS for RAG retrieval
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# ============================================================
# 1. Data Loading & Merging
# ============================================================

def build_merged_datasets(
    notes_path: str,
    train_path: str,
    test_path: str,
    out_train_merged: str,
    out_test_merged: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Always rebuild merged datasets from the original CSVs.
    Uses engine='python' to avoid C-engine buffer overflow issues.
    """
    print("[Data] Loading raw CSVs (engine='python')...")
    notes = pd.read_csv(notes_path, encoding="latin1", engine="python")
    train = pd.read_csv(train_path, encoding="latin1", engine="python")
    test = pd.read_csv(test_path, encoding="latin1", engine="python")

    # Fix BOM issues
    notes.columns = [c.replace("ï»¿", "") for c in notes.columns]

    print("[Data] Melting Notes into per-segment rows...")
    note_segments = notes.melt(
        id_vars=["Experiment", "Topic", "ID"],
        value_vars=["Segment1_Notes", "Segment2_Notes", "Segment3_Notes", "Segment4_Notes"],
        var_name="SegmentCol",
        value_name="NoteText",
    )
    note_segments["Segment"] = note_segments["SegmentCol"].str.extract(r"(\d)").astype(int)
    note_segments = note_segments.drop(columns=["SegmentCol"])

    print("[Data] Merging into train/test...")
    train_merged = train.merge(
        note_segments[["Topic", "ID", "Segment", "NoteText"]],
        on=["Topic", "ID", "Segment"],
        how="left",
    )
    test_merged = test.merge(
        note_segments[["Topic", "ID", "Segment", "NoteText"]],
        on=["Topic", "ID", "Segment"],
        how="left",
    )

    train_merged.to_csv(out_train_merged, index=False)
    test_merged.to_csv(out_test_merged, index=False)

    print(f"[Data] Saved {out_train_merged} ({len(train_merged)} rows)")
    print(f"[Data] Saved {out_test_merged} ({len(test_merged)} rows)")

    return train_merged, test_merged


# ============================================================
# 2. Rule-Based Model
# ============================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "being", "that", "this", "these", "those",
    "it", "as", "by", "with", "at", "from", "which", "who", "whom", "into",
    "about", "between", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "don", "should", "now",
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")

def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = TOKEN_PATTERN.findall(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

def overlap_features(idea: str, note: str) -> Tuple[float, float, float, int]:
    idea_tokens = tokenize(idea)
    note_tokens = tokenize(note)

    if not idea_tokens or not note_tokens:
        return 0.0, 0.0, 0.0, 0

    set_i = set(idea_tokens)
    set_n = set(note_tokens)
    inter = set_i & set_n
    union = set_i | set_n

    jacc = len(inter) / len(union) if union else 0.0
    recall_idea = len(inter) / len(set_i) if set_i else 0.0
    precision_note = len(inter) / len(set_n) if set_n else 0.0
    contains_all = int(set_i.issubset(set_n))
    return jacc, recall_idea, precision_note, contains_all

@dataclass
class RuleBasedModel:
    t_jacc: float = 0.3
    t_recall: float = 0.4
    t_prec: float = 0.0
    use_prec: bool = False

    def decision(self, idea: str, note: str) -> int:
        jacc, r_i, p_n, contains_all = overlap_features(idea, note)
        if contains_all:
            return 1
        if r_i >= self.t_recall:
            return 1
        if jacc >= self.t_jacc:
            return 1
        if self.use_prec and p_n >= self.t_prec:
            return 1
        return 0

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[RuleBased] Predict", leave=False):
            preds.append(self.decision(row["IdeaUnit"], row["NoteText"]))
        return np.array(preds)

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> None:
        y_true = df["label"].values
        best_f1 = -1.0
        best = None

        jacc_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        recall_values = [0.2, 0.3, 0.4, 0.5, 0.6]

        combos = [(tj, tr) for tj in jacc_values for tr in recall_values]
        for t_j, t_r in tqdm(combos, desc="[RuleBased] Grid search", leave=False):
            self.t_jacc = t_j
            self.t_recall = t_r
            preds = self.predict(df)
            f1 = f1_score(y_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best = (t_j, t_r)

        self.t_jacc, self.t_recall = best
        if verbose:
            print(f"[RuleBased] Best F1={best_f1:.4f} at t_jacc={self.t_jacc}, t_recall={self.t_recall}")


# ============================================================
# 3. BERT Embedding + Logistic Regression Models
# ============================================================

@dataclass
class BertLRModel:
    model_name: str
    clf: LogisticRegression = None
    encoder: SentenceTransformer = None

    def _load_encoder(self):
        if self.encoder is None:
            print(f"[BERT] Loading encoder: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)

    def _encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        self._load_encoder()
        return self.encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def fit(self, df: pd.DataFrame, C: float = 1.0, max_iter: int = 200, embed_bs: int = 32) -> None:
        print(f"[BERT-LR] Fitting model {self.model_name} on {len(df)} examples")
        ideas = df["IdeaUnit"].tolist()
        notes = df["NoteText"].tolist()
        y = df["label"].values

        h_I = self._encode_texts(ideas, batch_size=embed_bs)
        h_N = self._encode_texts(notes, batch_size=embed_bs)

        feats = np.concatenate([h_I, h_N, np.abs(h_I - h_N), h_I * h_N], axis=1)

        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(feats, y)
        self.clf = clf

        del h_I, h_N, feats
        gc.collect()

    def predict_proba(self, df: pd.DataFrame, embed_bs: int = 32) -> np.ndarray:
        assert self.clf is not None, "Model not fitted."
        ideas = df["IdeaUnit"].tolist()
        notes = df["NoteText"].tolist()

        h_I = self._encode_texts(ideas, batch_size=embed_bs)
        h_N = self._encode_texts(notes, batch_size=embed_bs)
        feats = np.concatenate([h_I, h_N, np.abs(h_I - h_N), h_I * h_N], axis=1)
        probs = self.clf.predict_proba(feats)[:, 1]

        del h_I, h_N, feats
        gc.collect()
        return probs

    def predict(self, df: pd.DataFrame, threshold: float = 0.5, embed_bs: int = 32) -> np.ndarray:
        probs = self.predict_proba(df, embed_bs=embed_bs)
        return (probs >= threshold).astype(int)


def train_multiple_bert_models(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    model_names: List[str],
    C: float = 1.0,
    max_iter: int = 200,
    embed_bs: int = 32,
    threshold: float = 0.5,
) -> Dict[str, BertLRModel]:
    models: Dict[str, BertLRModel] = {}
    for name in tqdm(model_names, desc="[BERT-LR] Training models"):
        model = BertLRModel(model_name=name)
        model.fit(df_train, C=C, max_iter=max_iter, embed_bs=embed_bs)

        y_true = df_valid["label"].values
        y_pred = model.predict(df_valid, threshold=threshold, embed_bs=embed_bs)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        print(f"[{name}] Valid F1={f1:.4f}, Acc={acc:.4f}")

        models[name] = model
    return models


def ensemble_predict(
    models: Dict[str, BertLRModel],
    df: pd.DataFrame,
    threshold: float = 0.5,
    embed_bs: int = 32,
) -> np.ndarray:
    probs_list = []
    for name, model in tqdm(models.items(), desc="[BERT-Ensemble] Predict", leave=False):
        probs = model.predict_proba(df, embed_bs=embed_bs)
        probs_list.append(probs)

    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    preds = (avg_probs >= threshold).astype(int)
    return preds


# ============================================================
# 4. LLM-based Helpers (plain + RAG)
# ============================================================

def majority_vote(predictions: List[int]) -> int:
    counts = Counter(predictions)
    if counts[0] > counts[1]:
        return 0
    elif counts[1] > counts[0]:
        return 1
    else:
        return 1  # tie-break -> 1 (as in your code)

def format_example_for_prompt(row: pd.Series) -> str:
    return (
        f"TOPIC: {row['Topic']}\n"
        f"IDEA UNIT: {row['IdeaUnit']}\n"
        f"NOTE SEGMENT: {row['NoteText']}\n"
        f"ANSWER: {row['label']}\n"
        "----\n"
    )

def build_fewshot_block(train_df: pd.DataFrame, topic: str, k: int = 6) -> str:
    topic_df = train_df[train_df["Topic"] == topic].copy()
    if topic_df.empty:
        topic_df = train_df.copy()

    pos = topic_df[topic_df["label"] == 1]
    neg = topic_df[topic_df["label"] == 0]

    examples = []
    half = max(1, k // 2)
    if len(pos) > 0:
        examples.extend(pos.sample(n=min(len(pos), half), random_state=42).to_dict("records"))
    if len(neg) > 0 and len(examples) < k:
        examples.extend(neg.sample(n=min(len(neg), k - len(examples)), random_state=43).to_dict("records"))

    random.shuffle(examples)
    text = ""
    for ex in examples:
        ex_row = pd.Series(ex)
        text += format_example_for_prompt(ex_row)
    return text

def build_plain_llm_prompt(row: pd.Series, train_df: pd.DataFrame, k: int = 6) -> str:
    topic = row["Topic"]
    fewshot = build_fewshot_block(train_df, topic, k=k)

    query = (
        "You are grading whether a student's note segment covers a specific idea from a lecture.\n"
        "For each case you see:\n"
        "- TOPIC\n"
        "- IDEA UNIT (a single statement from the lecturer)\n"
        "- NOTE SEGMENT (the student's notes for that segment)\n\n"
        "Your task: Output ONLY '1' if the note clearly covers the idea unit (explicitly or clearly paraphrased), "
        "and '0' otherwise. No extra words.\n\n"
        "Here are labeled examples:\n"
        "----\n"
        f"{fewshot}\n"
        "Now, classify the following case.\n\n"
        f"TOPIC: {row['Topic']}\n"
        f"IDEA UNIT: {row['IdeaUnit']}\n"
        f"NOTE SEGMENT: {row['NoteText']}\n"
        "ANSWER:"
    )
    return query

@dataclass
class RagIndex:
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder: SentenceTransformer = None
    index: Any = None
    vectors: np.ndarray = None
    meta: pd.DataFrame = None

    def _load_encoder(self):
        if self.encoder is None:
            print(f"[RAG] Loading encoder: {self.encoder_name}")
            self.encoder = SentenceTransformer(self.encoder_name)

    def build(self, train_df: pd.DataFrame) -> None:
        self._load_encoder()
        print("[RAG] Encoding training corpus...")
        corpus_texts = (
            train_df["IdeaUnit"].fillna("") + " [SEP] " +
            train_df["NoteText"].fillna("")
        ).tolist()
        self.vectors = self.encoder.encode(
            corpus_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
        self.meta = train_df.reset_index(drop=True)

        if HAS_FAISS:
            dim = self.vectors.shape[1]
            idx = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.vectors)
            idx.add(self.vectors)
            self.index = idx
        else:
            self.index = None
        print(f"[RAG] Built index with {len(self.meta)} entries (FAISS={HAS_FAISS})")

    def query(self, row: pd.Series, top_k: int = 5) -> pd.DataFrame:
        self._load_encoder()
        query_text = f"{row['IdeaUnit']} [SEP] {row['NoteText']}"
        q_vec = self.encoder.encode([query_text], convert_to_numpy=True).astype("float32")

        if self.index is not None:
            faiss.normalize_L2(q_vec)
            _, I = self.index.search(q_vec, top_k)
            indices = I[0]
        else:
            vecs = self.vectors
            vnorm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            qnorm = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-8
            sims = (vecs / vnorm) @ (q_vec.T / qnorm)
            sims = sims.squeeze(-1)
            indices = np.argsort(-sims)[:top_k]

        return self.meta.iloc[indices]

def build_rag_block(neighbors: pd.DataFrame) -> str:
    lines = []
    for _, ex in neighbors.iterrows():
        lines.append(
            "RETRIEVED EXAMPLE:\n"
            f"- TOPIC: {ex['Topic']}\n"
            f"- IDEA UNIT: {ex['IdeaUnit']}\n"
            f"- NOTE SEGMENT: {ex['NoteText']}\n"
            f"- LABEL: {ex['label']}\n"
            "----"
        )
    return "\n".join(lines)

def build_rag_llm_prompt(
    row: pd.Series,
    train_df: pd.DataFrame,
    rag_index: RagIndex,
    fewshot_k: int = 4,
    top_k: int = 4,
) -> str:
    topic = row["Topic"]
    fewshot = build_fewshot_block(train_df, topic, k=fewshot_k)
    neighbors = rag_index.query(row, top_k=top_k)
    rag_block = build_rag_block(neighbors)

    query = (
        "You are grading whether a student's note segment covers a specific idea from a lecture.\n"
        "You will see some labeled training examples and retrieved similar cases. Use them as guidance.\n\n"
        "TASK: Output ONLY '1' if the note clearly covers the idea unit (explicitly or clearly paraphrased), "
        "and '0' otherwise. Do not explain your reasoning.\n\n"
        "Labeled examples:\n"
        "----\n"
        f"{fewshot}\n\n"
        "Retrieved similar cases:\n"
        "----\n"
        f"{rag_block}\n\n"
        "Now classify the following case.\n"
        f"TOPIC: {row['Topic']}\n"
        f"IDEA UNIT: {row['IdeaUnit']}\n"
        f"NOTE SEGMENT: {row['NoteText']}\n"
        "ANSWER:"
    )
    return query

def extract_label_from_output(text: str) -> int:
    text = (text or "").strip()
    m = re.search(r"[01]", text)
    if m:
        return int(m.group(0))
    return 0


# ============================================================
# 5. Reporting Utilities
# ============================================================

def evaluate_split(
    name: str,
    split_name: str,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert len(df) == len(y_true) == len(y_pred), "Length mismatch in evaluate_split"

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    overall_df = pd.DataFrame([{
        "model": name,
        "split": split_name,
        "accuracy": acc,
        "f1": f1,
    }])

    df_reset = df.reset_index(drop=True)
    topic_rows = []
    for topic in df_reset["Topic"].unique():
        mask = df_reset["Topic"] == topic
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) == 0:
            continue
        topic_rows.append({
            "model": name,
            "split": split_name,
            "topic": topic,
            "accuracy": accuracy_score(yt, yp),
            "f1": f1_score(yt, yp),
        })

    topic_df = pd.DataFrame(topic_rows)
    return overall_df, topic_df

def write_report(
    overall_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    path_overall = os.path.join(out_dir, "metrics_overall.csv")
    path_topic = os.path.join(out_dir, "metrics_by_topic.csv")
    path_txt = os.path.join(out_dir, "metrics_report.txt")

    overall_df.to_csv(path_overall, index=False)
    topic_df.to_csv(path_topic, index=False)

    lines = []
    lines.append("=== Overall Metrics ===")
    for _, row in overall_df.sort_values(["split", "model"]).iterrows():
        lines.append(f"[{row['split']}] {row['model']}: Acc={row['accuracy']:.4f}, F1={row['f1']:.4f}")

    lines.append("\n=== Per-Topic Metrics ===")
    for (model, split), sub in topic_df.groupby(["model", "split"]):
        lines.append(f"\nModel: {model} | Split: {split}")
        for _, r in sub.iterrows():
            lines.append(f"  Topic={r['topic']}: Acc={r['accuracy']:.4f}, F1={r['f1']:.4f}")

    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Report] Saved {path_overall}, {path_topic}, {path_txt}")


# ============================================================
# 6. CLI + Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Note–IdeaUnit coverage pipeline (Rule/BERT/LLM/RAG)")

    # Inputs
    p.add_argument("--notes", default="Notes.csv", help="Path to Notes.csv")
    p.add_argument("--train", default="train.csv", help="Path to train.csv")
    p.add_argument("--test", default="test.csv", help="Path to test.csv")

    # Outputs
    p.add_argument("--out-dir", default="outputs", help="Output directory")
    p.add_argument("--train-merged", default="train_merged.csv", help="Output merged train CSV")
    p.add_argument("--test-merged", default="test_merged.csv", help="Output merged test CSV")

    # Split
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--train-frac", type=float, default=0.8, help="Train fraction for train/valid split")

    # Components toggles
    p.add_argument("--run-rule", action="store_true", help="Run rule-based baseline")
    p.add_argument("--run-bert", action="store_true", help="Run BERT-LR models + ensemble")
    p.add_argument("--run-llm", action="store_true", help="Run LLM (plain + RAG). Slow.")
    p.add_argument("--skip-merge", action="store_true", help="Skip rebuilding merged CSVs, read from --train-merged/--test-merged")

    # BERT params
    p.add_argument("--bert-models", nargs="+",
                   default=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
                   help="SentenceTransformer model names")
    p.add_argument("--bert-threshold", type=float, default=0.5, help="Threshold for BERT predictions")
    p.add_argument("--bert-embed-bs", type=int, default=32, help="Embedding batch size")
    p.add_argument("--bert-C", type=float, default=1.0, help="LR regularization strength C")
    p.add_argument("--bert-max-iter", type=int, default=200, help="LR max_iter")

    # LLM params
    p.add_argument("--llm-model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="LLM model path/name for lmdeploy")
    p.add_argument("--llm-max-num-seqs", type=int, default=4, help="Number of stochastic samples per instance")
    p.add_argument("--llm-max-model-len", type=int, default=1024, help="Base model length")
    p.add_argument("--llm-extra-context", type=int, default=1024, help="Extra context added to session_len")
    p.add_argument("--llm-max-new-tokens", type=int, default=32, help="Max new tokens per generation")

    # Decoding
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--min-p", type=float, default=0.01)

    # RAG params
    p.add_argument("--rag-encoder", default="sentence-transformers/all-MiniLM-L6-v2", help="RAG encoder")
    p.add_argument("--rag-top-k", type=int, default=4, help="Top-k retrieved examples")
    p.add_argument("--rag-fewshot-k", type=int, default=4, help="Few-shot examples in RAG prompt")
    p.add_argument("--plain-fewshot-k", type=int, default=6, help="Few-shot examples in plain prompt")

    return p.parse_args()


def main():
    args = parse_args()

    # Environment defaults
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Build merged datasets (or read existing)
    if args.skip_merge:
        print("[Data] Skipping merge. Reading merged CSVs...")
        train_merged = pd.read_csv(args.train_merged, encoding="latin1", engine="python")
        test_merged = pd.read_csv(args.test_merged, encoding="latin1", engine="python")
    else:
        train_merged, test_merged = build_merged_datasets(
            notes_path=args.notes,
            train_path=args.train,
            test_path=args.test,
            out_train_merged=args.train_merged,
            out_test_merged=args.test_merged,
        )

    # Clean train
    train_merged = train_merged.dropna(subset=["NoteText", "label"]).reset_index(drop=True)
    train_merged["label"] = train_merged["label"].astype(int)

    # Clean test
    test_merged = test_merged.dropna(subset=["NoteText"]).reset_index(drop=True)
    has_test_labels = False
    test_labeled = None
    if "label" in test_merged.columns:
        mask_labeled = test_merged["label"].notna()
        has_test_labels = bool(mask_labeled.any())
        if has_test_labels:
            test_labeled = test_merged[mask_labeled].reset_index(drop=True)
            test_labeled["label"] = test_labeled["label"].astype(int)

    # Train/valid split
    np.random.seed(args.seed)
    perm = np.random.permutation(len(train_merged))
    split = int(args.train_frac * len(train_merged))
    train_idx, valid_idx = perm[:split], perm[split:]
    df_train = train_merged.iloc[train_idx].reset_index(drop=True)
    df_valid = train_merged.iloc[valid_idx].reset_index(drop=True)

    y_valid = df_valid["label"].values
    y_test = test_labeled["label"].values if has_test_labels else None

    overall_entries: List[pd.DataFrame] = []
    topic_entries: List[pd.DataFrame] = []

    # ---------------- Rule-based
    if args.run_rule:
        rb = RuleBasedModel()
        rb.fit(df_train, verbose=True)
        y_valid_rb = rb.predict(df_valid)
        print("[RuleBased] Valid F1:", f1_score(y_valid, y_valid_rb))

        ov, tp = evaluate_split("RuleBased", "valid", df_valid, y_valid, y_valid_rb)
        overall_entries.append(ov); topic_entries.append(tp)

        if has_test_labels:
            y_test_rb = rb.predict(test_labeled)
            print("[RuleBased] Test F1:", f1_score(y_test, y_test_rb))
            ov, tp = evaluate_split("RuleBased", "test", test_labeled, y_test, y_test_rb)
            overall_entries.append(ov); topic_entries.append(tp)

    # ---------------- BERT-LR + ensemble
    bert_models: Optional[Dict[str, BertLRModel]] = None
    if args.run_bert:
        bert_models = train_multiple_bert_models(
            df_train=df_train,
            df_valid=df_valid,
            model_names=args.bert_models,
            C=args.bert_C,
            max_iter=args.bert_max_iter,
            embed_bs=args.bert_embed_bs,
            threshold=args.bert_threshold,
        )

        for name, model in bert_models.items():
            y_valid_pred = model.predict(df_valid, threshold=args.bert_threshold, embed_bs=args.bert_embed_bs)
            ov, tp = evaluate_split(f"BERT-LR:{name}", "valid", df_valid, y_valid, y_valid_pred)
            overall_entries.append(ov); topic_entries.append(tp)

            if has_test_labels:
                y_test_pred = model.predict(test_labeled, threshold=args.bert_threshold, embed_bs=args.bert_embed_bs)
                ov, tp = evaluate_split(f"BERT-LR:{name}", "test", test_labeled, y_test, y_test_pred)
                overall_entries.append(ov); topic_entries.append(tp)

        # Ensemble
        y_valid_ens = ensemble_predict(bert_models, df_valid, threshold=args.bert_threshold, embed_bs=args.bert_embed_bs)
        print("[BERT-Ensemble] Valid F1:", f1_score(y_valid, y_valid_ens))
        ov, tp = evaluate_split("BERT-Ensemble", "valid", df_valid, y_valid, y_valid_ens)
        overall_entries.append(ov); topic_entries.append(tp)

        if has_test_labels:
            y_test_ens = ensemble_predict(bert_models, test_labeled, threshold=args.bert_threshold, embed_bs=args.bert_embed_bs)
            print("[BERT-Ensemble] Test F1:", f1_score(y_test, y_test_ens))
            ov, tp = evaluate_split("BERT-Ensemble", "test", test_labeled, y_test, y_test_ens)
            overall_entries.append(ov); topic_entries.append(tp)

        # Save BERT ensemble predictions for ALL test rows
        print("[BERT-Ensemble] Predicting on full test_merged...")
        test_preds_bert_full = ensemble_predict(bert_models, test_merged, threshold=args.bert_threshold, embed_bs=args.bert_embed_bs)
        sub = test_merged.copy()
        sub["pred_label_bert_ens"] = test_preds_bert_full
        os.makedirs(args.out_dir, exist_ok=True)
        out_sub = os.path.join(args.out_dir, "submission_bert_ensemble.csv")
        sub[["Experiment", "Topic", "ID", "Segment", "IdeaUnit", "pred_label_bert_ens"]].to_csv(out_sub, index=False)
        print(f"[Output] Saved {out_sub}")

    # ---------------- LLM (plain + RAG)
    if args.run_llm:
        from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

        # LLM engine
        llm_context = args.llm_max_model_len + args.llm_extra_context
        llm_pipe = pipeline(
            args.llm_model,
            backend_config=TurbomindEngineConfig(
                cache_max_entry_count=0.90,
                session_len=llm_context,
                tp=1
            ),
            stream_response=False,
        )

        def run_llm_batch(prompts: List[str], max_tokens: int) -> List[str]:
            gen_config = GenerationConfig(
                temperature=args.temperature,
                min_p=args.min_p,
                top_p=args.top_p,
                skip_special_tokens=True,
                max_new_tokens=max_tokens,
                do_sample=True,
            )
            response = llm_pipe(prompts=prompts, gen_config=gen_config)
            return [out.text for out in response]

        def predict_llm_for_row(
            row: pd.Series,
            train_df: pd.DataFrame,
            rag_index: Optional[RagIndex],
            use_rag: bool,
        ) -> int:
            prompts: List[str] = []
            for _ in range(args.llm_max_num_seqs):
                if use_rag and rag_index is not None:
                    p = build_rag_llm_prompt(
                        row=row,
                        train_df=train_df,
                        rag_index=rag_index,
                        fewshot_k=args.rag_fewshot_k,
                        top_k=args.rag_top_k,
                    )
                else:
                    p = build_plain_llm_prompt(row=row, train_df=train_df, k=args.plain_fewshot_k)
                prompts.append(p)

            raw_outputs = run_llm_batch(prompts, max_tokens=args.llm_max_new_tokens)
            outs = [extract_label_from_output(txt) for txt in raw_outputs]
            if not outs:
                outs = [0]
            return majority_vote(outs)

        # RAG index
        rag_index = RagIndex(encoder_name=args.rag_encoder)
        rag_index.build(train_merged)

        # LLM Plain on valid (+ labeled test)
        print("[LLM-Plain] Predicting on valid...")
        y_valid_llm_plain = []
        for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="[LLM-Plain] Valid"):
            y_valid_llm_plain.append(predict_llm_for_row(row, train_merged, rag_index=None, use_rag=False))
        y_valid_llm_plain = np.array(y_valid_llm_plain)

        ov, tp = evaluate_split("LLM-Plain", "valid", df_valid, y_valid, y_valid_llm_plain)
        overall_entries.append(ov); topic_entries.append(tp)

        if has_test_labels:
            print("[LLM-Plain] Predicting on test (labeled subset)...")
            y_test_llm_plain = []
            for _, row in tqdm(test_labeled.iterrows(), total=len(test_labeled), desc="[LLM-Plain] Test"):
                y_test_llm_plain.append(predict_llm_for_row(row, train_merged, rag_index=None, use_rag=False))
            y_test_llm_plain = np.array(y_test_llm_plain)

            ov, tp = evaluate_split("LLM-Plain", "test", test_labeled, y_test, y_test_llm_plain)
            overall_entries.append(ov); topic_entries.append(tp)

        # LLM-RAG on valid (+ labeled test)
        print("[LLM-RAG] Predicting on valid...")
        y_valid_llm_rag = []
        for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="[LLM-RAG] Valid"):
            y_valid_llm_rag.append(predict_llm_for_row(row, train_merged, rag_index=rag_index, use_rag=True))
        y_valid_llm_rag = np.array(y_valid_llm_rag)

        ov, tp = evaluate_split("LLM-RAG", "valid", df_valid, y_valid, y_valid_llm_rag)
        overall_entries.append(ov); topic_entries.append(tp)

        if has_test_labels:
            print("[LLM-RAG] Predicting on test (labeled subset)...")
            y_test_llm_rag = []
            for _, row in tqdm(test_labeled.iterrows(), total=len(test_labeled), desc="[LLM-RAG] Test"):
                y_test_llm_rag.append(predict_llm_for_row(row, train_merged, rag_index=rag_index, use_rag=True))
            y_test_llm_rag = np.array(y_test_llm_rag)

            ov, tp = evaluate_split("LLM-RAG", "test", test_labeled, y_test, y_test_llm_rag)
            overall_entries.append(ov); topic_entries.append(tp)

    # ---------------- Write reports (if anything ran)
    if overall_entries:
        overall_df = pd.concat(overall_entries, ignore_index=True)
        topic_df = pd.concat(topic_entries, ignore_index=True)
        write_report(overall_df, topic_df, out_dir=args.out_dir)
    else:
        print("[Warn] No components were run. Use --run-rule / --run-bert / --run-llm.")


if __name__ == "__main__":
    main()
