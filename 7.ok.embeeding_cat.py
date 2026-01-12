import os
import re
import json
import joblib
import time
import gc  # Garbage Collector for memory management
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import GroupShuffleSplit

# =========================
# CONFIG
# =========================

DATA_PATH = "ok.clean_vuln_dataset_imputed.parquet"
MODEL_NAME = "microsoft/codebert-base"
OUTPUT_DIR = "models/xgb_codebert_hybrid_ab_shap"

# Save paths
EMBEDDING_PATH_TRAIN = os.path.join(OUTPUT_DIR, "X_train_emb.npy")
EMBEDDING_PATH_TEST = os.path.join(OUTPUT_DIR, "X_test_emb.npy")
META_PATH_TRAIN = os.path.join(OUTPUT_DIR, "df_train_meta.pkl")
META_PATH_TEST = os.path.join(OUTPUT_DIR, "df_test_meta.pkl")
LABELS_PATH_TRAIN = os.path.join(OUTPUT_DIR, "y_train.csv")
LABELS_PATH_TEST = os.path.join(OUTPUT_DIR, "y_test.csv")
NAMES_PATH = os.path.join(OUTPUT_DIR, "meta_feature_names.json")

# Sampling control (use full dataset if N_SAMPLES is larger than dataset size)
N_SAMPLES = 50_000  # 2_000_000

MAX_LEN = 256
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42
CWE_MIN_COUNT = 50

# =========================
# HELPERS
# =========================

# Regex for extracting CVE year/number components
CVE_REGEX = re.compile(r"CVE-(\d{4})-(\d+)", re.IGNORECASE)

# Lightweight tokenizer-like regex for estimating token counts in code
TOKEN_REGEX = re.compile(r"[A-Za-z_]\w+|\d+|==|!=|<=|>=|[^\s]", re.MULTILINE)


def ensure_dir(path: str):
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def extract_cve_number(cve_id: str) -> float:
    """Extract the numeric part of CVE-YYYY-NNNN... as float; returns NaN if invalid."""
    if not isinstance(cve_id, str):
        return np.nan
    m = CVE_REGEX.search(cve_id)
    return float(m.group(2)) if m else np.nan


def code_token_count(code: str) -> int:
    """Estimate number of tokens in code using a regex-based tokenization heuristic."""
    if not isinstance(code, str) or not code:
        return 0
    return len(TOKEN_REGEX.findall(code))


def print_balance(title: str, y: pd.Series):
    """Print class balance for a label series."""
    vc = y.value_counts(dropna=False)
    print(f"\n=== BALANCE REPORT: {title} ===\n{vc}")


def batch_iterable(lst: List[str], batch_size: int):
    """Yield list chunks of size batch_size."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def embed_batch(texts: List[str], tokenizer, model, device, max_len: int) -> np.ndarray:
    """
    Compute CodeBERT embeddings for a batch of texts using mean pooling over token embeddings.
    Returns numpy array of shape: (batch_size, hidden_dim).
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model(**enc)
        hidden = out.last_hidden_state  # (B, T, H)

        # Mean pooling using attention mask to ignore padding tokens
        mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return emb.cpu().numpy()


# Incremental embedding writer: writes embeddings directly to disk using memmap
def embed_series_to_disk(
    series: pd.Series,
    tokenizer,
    model,
    device,
    max_len: int,
    batch_size: int,
    filename: str
) -> np.ndarray:
    """
    Embed a pandas Series of code strings and write the resulting embeddings incrementally to disk.

    Why memmap:
    - Avoid holding the full embedding matrix in RAM
    - Allows very large datasets to be processed safely

    Returns:
    - np.memmap pointing to the on-disk array (shape: [N, 768])
    """
    start_time = time.time()
    texts = series.astype(str).tolist()
    total = len(texts)

    # CodeBERT-base hidden size
    num_features = 768

    print(f"[EMBED] Starting incremental embedding of {total} samples to {filename}...")

    # Create memmap array on disk (float32 saves space)
    fp = np.memmap(filename, dtype='float32', mode='w+', shape=(total, num_features))

    done = 0
    for batch in batch_iterable(texts, batch_size):
        emb_batch = embed_batch(batch, tokenizer, model, device, max_len)

        # Write batch embedding into correct slice
        fp[done: done + len(batch), :] = emb_batch
        done += len(batch)

        # Periodic progress logging + disk flush
        if done % (batch_size * 200) == 0 or done == total:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            print(f"[EMBED] Progress: {done}/{total} | Rate: {rate:.1f} s/s | Disk-Syncing...")
            fp.flush()

    print(f"[EMBED] Finished! Saved to {filename}. Total time: {time.time() - start_time:.1f}s")
    return fp


def build_meta_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build non-embedding (tabular) features:
    - One-hot encodings for: language, dataset, and a slimmed CWE category
    - Numeric features: token count, CVE number, code length/lines, CVE year stats, duplication counts
    """
    df = df.copy()

    # Create a reduced CWE category to control cardinality
    if "cwe_id" in df.columns:
        cwe_counts = df["cwe_id"].astype(str).value_counts()
        common = set(cwe_counts[cwe_counts >= CWE_MIN_COUNT].index)
        df["cwe_id_slim"] = df["cwe_id"].astype(str).where(
            df["cwe_id"].astype(str).isin(common),
            "CWE_OTHER"
        )
    else:
        df["cwe_id_slim"] = "CWE_OTHER"

    # One-hot encode categorical features
    cat_df = df[["language", "dataset", "cwe_id_slim"]].copy()
    X_cat = pd.get_dummies(cat_df, dummy_na=False)

    # Numeric engineered features
    df["code_token_count"] = df["code"].apply(code_token_count)
    df["cve_number"] = df["cve_id"].apply(extract_cve_number)

    X_num = df[
        [
            "code_token_count",
            "cve_number",
            "code_len",
            "code_lines",
            "cve_year",
            "year_count",
            "code_dup_count",
            "code_year_dup_count",
        ]
    ].copy()

    # Final meta feature matrix
    X_meta = pd.concat([X_cat, X_num], axis=1)
    return X_meta, list(X_meta.columns)


# =========================
# MAIN
# =========================

def main_embed():
    """Main pipeline: load -> meta features -> group split -> embed -> save artifacts."""
    ensure_dir(OUTPUT_DIR)

    # 1) Load dataset
    print(f"[LOAD] Reading data from {DATA_PATH}...")
    df_full = pd.read_parquet(DATA_PATH)

    # Optional sampling for faster iteration
    if len(df_full) > N_SAMPLES:
        df_full = df_full.sample(N_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)

    # 2) Build meta features before split to keep consistent one-hot columns
    X_meta_all, meta_feature_names = build_meta_features(df_full)

    # 3) Group split by CVE ID to avoid leakage of same CVE across train/test
    print(f"[SPLIT] GroupShuffleSplit by cve_id...")
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df_full, groups=df_full["cve_id"]))

    # Create code and label series
    code_train = df_full.iloc[train_idx]["code"].reset_index(drop=True)
    code_test = df_full.iloc[test_idx]["code"].reset_index(drop=True)
    y_train = df_full.iloc[train_idx]["label"].reset_index(drop=True)
    y_test = df_full.iloc[test_idx]["label"].reset_index(drop=True)

    # Split meta feature matrices
    df_train_meta = X_meta_all.iloc[train_idx].reset_index(drop=True)
    df_test_meta = X_meta_all.iloc[test_idx].reset_index(drop=True)

    # Free memory from large objects no longer needed
    del df_full, X_meta_all
    gc.collect()

    # 4) Load CodeBERT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MODEL] Device: {device}. Loading CodeBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    base_model.eval()

    # 5) Compute embeddings with incremental disk write
    print("\n--- Starting TRAIN Embedding ---")
    embed_series_to_disk(
        code_train, tokenizer, base_model, device,
        MAX_LEN, BATCH_SIZE, EMBEDDING_PATH_TRAIN
    )

    # Reduce memory before embedding test set
    del code_train
    gc.collect()

    print("\n--- Starting TEST Embedding ---")
    embed_series_to_disk(
        code_test, tokenizer, base_model, device,
        MAX_LEN, BATCH_SIZE, EMBEDDING_PATH_TEST
    )

    # Cleanup model and remaining large objects
    del code_test, base_model
    gc.collect()

    # 6) Save meta and label artifacts
    print("\n[SAVE] Saving Meta features and Labels...")
    df_train_meta.to_pickle(META_PATH_TRAIN)
    df_test_meta.to_pickle(META_PATH_TEST)
    y_train.to_csv(LABELS_PATH_TRAIN, index=False)
    y_test.to_csv(LABELS_PATH_TEST, index=False)

    # Save meta feature names for later alignment
    with open(NAMES_PATH, 'w') as f:
        json.dump(meta_feature_names, f)

    print(f"\n[SUCCESS] All data saved to {OUTPUT_DIR}. No memory crash!")


if __name__ == "__main__":
    main_embed()
