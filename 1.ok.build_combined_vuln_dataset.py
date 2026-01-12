"""
build_combined_vuln_dataset.py

Script to load multiple public vulnerability datasets
(MegaVul, BigVul, DiverseVul from HuggingFace + Juliet locally),
normalize them into a unified schema and save as CSV/Parquet.

Core columns:
- code:     source code snippet (function or file)
- label:    1 = vulnerable, 0 = non-vulnerable
- language: short language string (e.g. 'c', 'cpp', 'java', 'py', 'js', ...)
- dataset:  source dataset id ('megavul', 'bigvul', 'diversevul', 'juliet')

PLUS:
- cve_id:   unified CVE identifier if present in original metadata (else NaN)
- cwe_id:   unified CWE identifier if present (or extracted, e.g. from Juliet)
"""

from pathlib import Path
from typing import List

import re
import pandas as pd
from datasets import load_dataset


# =========================
# Helpers
# =========================

# Regex to detect CWE identifiers in various formats (e.g., CWE-79, CWE_121, cwe121)
CWE_REGEX = re.compile(r"(CWE)[-_]?(\d+)", re.IGNORECASE)


def extract_cwe_from_text(text: str) -> str | None:
    """
    Extract CWE id from a string and normalize it to 'CWE-<number>'.

    Examples:
    - 'CWE121_Stack_Based_Buffer_Overflow__...' -> 'CWE-121'
    - 'cwe-79' -> 'CWE-79'
    """
    # Guard against non-string inputs
    if not isinstance(text, str):
        return None

    # Search for CWE pattern
    m = CWE_REGEX.search(text)
    if not m:
        return None

    # Return normalized CWE format
    return f"CWE-{m.group(2)}"


def add_cve_cwe_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Add unified 'cve_id' and 'cwe_id' columns to the DataFrame.

    Logic:
    - Detect any existing columns whose names contain 'cve' or 'cwe'
    - Merge them into single canonical columns
    - Normalize CWE values when possible using regex
    - Preserve all original metadata columns unchanged
    """
    # Work on a copy to avoid mutating input DataFrame
    df = df.copy()

    # Detect candidate CVE/CWE columns by name
    cve_cols = [c for c in df.columns if "cve" in c.lower()]
    cwe_cols = [c for c in df.columns if "cwe" in c.lower()]

    # Helper to merge multiple metadata columns into one
    def _combine_cols(cols: list[str]) -> pd.Series:
        # No candidate columns -> return NA series
        if not cols:
            return pd.Series(pd.NA, index=df.index)

        # Single column -> clean empty/null-like values
        if len(cols) == 1:
            s = df[cols[0]].astype(str).str.strip()
            s = s.mask(s.str.lower().isin(["", "nan", "none", "null"]))
            return s

        # Multiple columns -> take first non-null per row
        tmp = df[cols].astype(str)
        tmp = tmp.apply(lambda col: col.str.strip())
        tmp = tmp.replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "None": pd.NA,
                "none": pd.NA,
                "null": pd.NA,
            }
        )
        merged = tmp.bfill(axis=1).iloc[:, 0]
        return merged

    # Build unified CVE and CWE series
    cve_series = _combine_cols(cve_cols)
    cwe_series = _combine_cols(cwe_cols)

    # Normalize CWE values using regex when possible
    cwe_series = cwe_series.astype(object)
    for idx, val in cwe_series.items():
        if pd.isna(val):
            continue
        cwe_normalized = extract_cwe_from_text(str(val))
        if cwe_normalized:
            cwe_series.loc[idx] = cwe_normalized
        else:
            # Keep original value if it cannot be parsed
            cwe_series.loc[idx] = str(val).strip()

    # Attach unified columns
    df["cve_id"] = cve_series
    df["cwe_id"] = cwe_series

    # Log detected metadata columns
    print(f"[{dataset_name}] CVE columns candidates: {cve_cols}")
    print(f"[{dataset_name}] CWE columns candidates: {cwe_cols}")

    return df


# =========================
# 1. MegaVul (HF: hitoshura25/megavul)
# =========================

def load_megavul(split: str = "train") -> pd.DataFrame:
    """
    Load MegaVul from HuggingFace and expand it into
    vulnerable (label=1) and fixed (label=0) samples.

    All original metadata columns are preserved.
    """
    print("[MegaVul] Loading from HuggingFace...")
    ds = load_dataset("hitoshura25/megavul", split=split)
    df = ds.to_pandas()

    # Normalize language column if present
    if "language" in df.columns:
        df["language"] = df["language"].astype(str).str.strip().str.lower()
    else:
        df["language"] = "unknown"

    # Metadata columns exclude the two code variants
    meta_cols = [c for c in df.columns if c not in ["vulnerable_code", "fixed_code"]]

    # Vulnerable samples
    vuln = df[["vulnerable_code"] + meta_cols].copy()
    vuln = vuln.rename(columns={"vulnerable_code": "code"})
    vuln["label"] = 1
    vuln["dataset"] = "megavul"

    # Fixed (non-vulnerable) samples
    fixed = df[["fixed_code"] + meta_cols].copy()
    fixed = fixed.rename(columns={"fixed_code": "code"})
    fixed["label"] = 0
    fixed["dataset"] = "megavul"

    # Combine vulnerable and fixed rows
    all_df = pd.concat([vuln, fixed], ignore_index=True)

    # Basic cleaning
    all_df["code"] = all_df["code"].astype(str)
    all_df = all_df[all_df["code"].str.strip() != ""]
    all_df["label"] = all_df["label"].astype(int)

    # Add unified CVE/CWE columns
    all_df = add_cve_cwe_columns(all_df, "MegaVul")

    print(f"[MegaVul] Total rows after expansion: {len(all_df)}")
    return all_df


# =========================
# 2. BigVul (HF: bstee615/bigvul)
# =========================

def load_bigvul(split: str = "train") -> pd.DataFrame:
    """
    Load BigVul from HuggingFace.

    Uses:
    - func_before as code
    - vul as label
    - lang as language
    """
    print("[BigVul] Loading from HuggingFace...")
    ds = load_dataset("bstee615/bigvul", split=split)
    df = ds.to_pandas()

    # Preserve all metadata except core fields
    meta_cols = [c for c in df.columns if c not in ["func_before", "func_after", "vul", "lang"]]

    df = df[["func_before", "vul", "lang"] + meta_cols].copy()
    df = df.rename(
        columns={
            "func_before": "code",
            "vul": "label",
            "lang": "language",
        }
    )

    # Basic cleaning and normalization
    df["code"] = df["code"].astype(str)
    df = df[df["code"].str.strip() != ""]
    df["label"] = df["label"].astype(int)
    df["language"] = df["language"].astype(str).str.strip().str.lower()
    df["dataset"] = "bigvul"

    # Add unified CVE/CWE columns
    df = add_cve_cwe_columns(df, "BigVul")

    print(f"[BigVul] Total rows: {len(df)}")
    return df


# =========================
# 3. DiverseVul (HF: bstee615/diversevul)
# =========================

def load_diversevul(split: str = "train") -> pd.DataFrame:
    """
    Load DiverseVul from HuggingFace.

    If loading fails or schema is unexpected,
    return an empty DataFrame and continue pipeline.
    """
    print("[DiverseVul] Loading from HuggingFace...")

    try:
        ds = load_dataset("bstee615/diversevul", split=split)
    except Exception as e:
        print(f"[DiverseVul] WARNING: failed to load dataset: {e}")
        print("[DiverseVul] Continuing WITHOUT DiverseVul.")
        return pd.DataFrame()

    df = ds.to_pandas()

    # Validate expected columns
    if not {"func", "target"}.issubset(df.columns):
        print("[DiverseVul] WARNING: expected columns 'func' and 'target' not found.")
        print("[DiverseVul] Columns found:", list(df.columns))
        print("[DiverseVul] Skipping DiverseVul.")
        return pd.DataFrame()

    meta_cols = [c for c in df.columns if c not in ["func", "target"]]

    df = df[["func", "target"] + meta_cols].copy()
    df = df.rename(columns={"func": "code", "target": "label"})

    # Cleaning and normalization
    df["code"] = df["code"].astype(str)
    df = df[df["code"].str.strip() != ""]
    df["label"] = df["label"].astype(int)
    df["language"] = "c/cpp"
    df["dataset"] = "diversevul"

    # Add unified CVE/CWE columns
    df = add_cve_cwe_columns(df, "DiverseVul")

    print(f"[DiverseVul] Total rows: {len(df)}")
    return df


# =========================
# 4. Juliet Test Suite (local files)
# =========================

def load_juliet(root_dir: str = "data/juliet") -> pd.DataFrame:
    """
    Load Juliet Test Suite from local filesystem.

    Labeling logic:
    - filenames containing 'bad'  -> vulnerable (1)
    - filenames containing 'good' -> non-vulnerable (0)

    No real CVE IDs exist; CWE is inferred from filename when possible.
    """
    root = Path(root_dir)
    if not root.exists():
        print(f"[Juliet] WARNING: root_dir {root_dir} does not exist.")
        return pd.DataFrame(columns=["code", "label", "language", "dataset", "cwe_id", "cve_id"])

    print(f"[Juliet] Scanning directory: {root_dir}")
    rows: List[dict] = []

    # Process C/C++ files
    for f in root.rglob("*.c"):
        name_lower = f.name.lower()
        if "bad" in name_lower:
            label = 1
        elif "good" in name_lower:
            label = 0
        else:
            continue

        try:
            code = f.read_text(errors="ignore")
        except Exception:
            continue
        if not code.strip():
            continue

        cwe_id = extract_cwe_from_text(f.name)

        rows.append(
            {
                "code": code,
                "label": label,
                "language": "c",
                "dataset": "juliet",
                "cwe_id": cwe_id,
                "cve_id": pd.NA,
            }
        )

    # Process Java files
    for f in root.rglob("*.java"):
        name_lower = f.name.lower()
        if "bad" in name_lower:
            label = 1
        elif "good" in name_lower:
            label = 0
        else:
            continue

        try:
            code = f.read_text(errors="ignore")
        except Exception:
            continue
        if not code.strip():
            continue

        cwe_id = extract_cwe_from_text(f.name)

        rows.append(
            {
                "code": code,
                "label": label,
                "language": "java",
                "dataset": "juliet",
                "cwe_id": cwe_id,
                "cve_id": pd.NA,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["label"] = df["label"].astype(int)

    print(f"[Juliet] Total rows: {len(df)}")
    if "cwe_id" in df.columns:
        print("[Juliet] CWE distribution (first 10):")
        print(df["cwe_id"].value_counts(dropna=False).head(10))

    return df


# =========================
# 5. Combine all
# =========================

def main():
    # Load individual datasets
    megavul_df = load_megavul(split="train")
    bigvul_df = load_bigvul(split="train")
    diversevul_df = load_diversevul(split="train")
    juliet_df = load_juliet(root_dir="data/juliet")

    # Collect available DataFrames
    dfs = [megavul_df, bigvul_df]
    if diversevul_df is not None and not diversevul_df.empty:
        dfs.append(diversevul_df)
    if juliet_df is not None and not juliet_df.empty:
        dfs.append(juliet_df)

    # Concatenate all datasets
    combined = pd.concat(dfs, ignore_index=True)

    # Final normalization and sanity checks
    combined["code"] = combined["code"].astype(str)
    combined = combined[combined["code"].str.strip() != ""]
    combined["label"] = combined["label"].astype(int)
    combined["language"] = combined["language"].astype(str).str.strip().str.lower()
    combined["dataset"] = combined["dataset"].astype(str)

    # Ensure CVE/CWE columns exist
    if "cve_id" not in combined.columns:
        combined["cve_id"] = pd.NA
    if "cwe_id" not in combined.columns:
        combined["cwe_id"] = pd.NA

    # Dataset summary
    print("\n=== Combined dataset summary ===")
    print("Total rows:", len(combined))
    print("By dataset:\n", combined["dataset"].value_counts())
    print("By language:\n", combined["language"].value_counts())

    print("\nColumns in combined dataset:")
    print(list(combined.columns))

    # Output files
    out_csv = "combined_vuln_dataset_with_meta.csv"
    out_parquet = "combined_vuln_dataset_with_meta.parquet"

    combined.to_csv(out_csv, index=False)
    try:
        combined.to_parquet(out_parquet, index=False)
    except Exception as e:
        print(f"Could not save Parquet ({e}), CSV is still saved.")

    print(f"\nSaved CSV to: {out_csv}")
    print(f"Saved Parquet to: {out_parquet}")


if __name__ == "__main__":
    main()
