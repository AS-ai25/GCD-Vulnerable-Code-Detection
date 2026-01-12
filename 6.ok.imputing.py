import pandas as pd
import numpy as np
from typing import List, Dict

# =========================
# CONFIG
# =========================

# Path to the enriched Parquet dataset created earlier
DATA_PATH = "ok.clean_vuln_dataset_full.parquet"

# Output 1: CSV for quick inspection/debugging
OUTPUT_CSV = "ok.clean_vuln_dataset_imputed.csv"

# Output 2: Parquet for efficiency in later stages (embeddings, XGBoost, etc.)
OUTPUT_PARQUET = "ok.clean_vuln_dataset_imputed.parquet"

# Required columns to load and keep in the pipeline
NEEDED_COLUMNS = {
    "code", "label", "language", "dataset", "code_len", "code_lines",
    "cve_year", "year_count", "code_dup_count", "code_year_dup_count",
    "cve_id",
    "cwe_id",
}

# Rows missing any of these columns will be dropped
REQUIRED_FOR_DROPNA = ["code", "label"]

# =========================
# IMPUTATION CONFIGURATION
# =========================

# Fill values for ID/categorical identifier columns
CAT_ID_IMPUTE: Dict[str, str] = {
    "cve_id": "CVE-UNKNOWN",
    "cwe_id": "CWE-NONE",
}

# Numeric columns where missing values will be filled with -1.0
NUMERIC_IMPUTE: List[str] = [
    "code_len",
    "code_lines",
    "cve_year",
    "year_count",
    "code_dup_count",
    "code_year_dup_count"
]


# =========================
# MAIN IMPUTATION FUNCTION
# =========================

def impute_dataset_optimized(df_path: str, out_csv: str, out_parquet: str):
    """
    Loads from Parquet, performs imputation/cleanup, and saves to both Parquet and CSV.
    """
    print(f"[LOAD] Reading dataset from {df_path}...")

    # Efficient Parquet read (only needed columns)
    try:
        df = pd.read_parquet(df_path, columns=list(NEEDED_COLUMNS))
        print(f"[LOAD] Successfully loaded {len(df)} rows and {len(df.columns)} required columns.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load Parquet: {e}")
        return

    total_rows_before = len(df)

    # 1) Basic cleanup + drop rows missing mandatory fields
    print(f"[CLEAN] Checking required columns: {REQUIRED_FOR_DROPNA}")
    df_cleaned = df.dropna(subset=REQUIRED_FOR_DROPNA)

    # Remove empty code strings and enforce label type
    df_cleaned["code"] = df_cleaned["code"].astype(str).str.strip()
    df_cleaned = df_cleaned[df_cleaned["code"] != ""]
    df_cleaned["label"] = df_cleaned["label"].astype(int)

    total_rows_after = len(df_cleaned)
    if total_rows_before != total_rows_after:
        print(f"[CLEAN] Dropped {total_rows_before - total_rows_after} rows where 'code' or 'label' was missing/empty.")

    # 2) Impute categorical / ID-like columns
    print("[IMPUTE] Handling Categorical/ID columns...")
    for col, value in CAT_ID_IMPUTE.items():
        if col in df_cleaned.columns:
            df_cleaned[col] = (
                df_cleaned[col]
                .astype(str)
                .replace(['None', 'nan', ''], np.nan)
                .fillna(value)
            )
            print(f"  -> {col}: Filled NaN with '{value}'")

    # 3) Impute numeric columns with -1.0
    print("[IMPUTE] Handling Numeric columns with -1.0...")
    for col in NUMERIC_IMPUTE:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(-1.0)
            print(f"  -> {col}: Filled NaN with -1.0")

    # 4) Final verification + fallback fills for common categorical columns
    print("\n[VERIFY] Checking remaining Nulls...")
    missing_after = df_cleaned[list(NEEDED_COLUMNS)].isnull().sum()
    missing_after = missing_after[missing_after > 0]

    if missing_after.empty:
        print("  -> SUCCESS: All required columns are now clean.")
    else:
        print("  -> WARNING: Some columns still have missing values after imputation:")
        print(missing_after)

        # Fallback for common categoricals not covered in CAT_ID_IMPUTE
        df_cleaned = df_cleaned.fillna({'language': 'UNKNOWN', 'dataset': 'UNKNOWN'})

    # 5) Save outputs: Parquet first (preferred), then CSV
    print(f"\n[SAVE] Writing imputed dataset to {out_parquet} (Parquet)")
    df_cleaned.to_parquet(out_parquet, index=False)

    print(f"[SAVE] Writing imputed dataset to {out_csv} (CSV)")
    df_cleaned.to_csv(out_csv, index=False)

    print(f"[DONE] Imputation complete. Total rows: {len(df_cleaned)}")


if __name__ == "__main__":
    impute_dataset_optimized(DATA_PATH, OUTPUT_CSV, OUTPUT_PARQUET)
