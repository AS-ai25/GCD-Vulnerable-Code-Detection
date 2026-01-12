import os
import re
import time
import json
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

# =========================
# CONFIG
# =========================

# Input cleaned dataset path (must include: code, cve_id, cwe_id)
INPUT_CSV = "ok.clean_vuln_dataset.csv"

# Output enriched dataset path (adds CWE descriptions + derived features)
OUTPUT_CSV = "ok.clean_vuln_dataset_enriched.csv"

# MITRE CWE definition page template (CWE number will be formatted into {})
MITRE_CWE_URL = "https://cwe.mitre.org/data/definitions/{}.html"

# HTTP headers to reduce blocking / mimic a normal browser request
HEADERS = {"User-Agent": "Mozilla/5.0 (CWE-fetcher; Data-Enrichment-Script)"}

# Rate limiting between requests to MITRE to be polite and reduce throttling risk
SLEEP_BETWEEN_REQUESTS = 1.0

# Local JSON cache file to avoid refetching already-seen CWE descriptions
CACHE_FILE = "cwe_cache.json"

# CVE format validator/parser: CVE-YYYY-NNNN...
CVE_RE = re.compile(r"^CVE-(\d{4})-(\d+)$", re.IGNORECASE)


# =========================
# CACHING HELPERS
# =========================

def load_cwe_cache(path: str) -> Dict[str, str | None]:
    """Loads an existing CWE cache from a JSON file (if present)."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                print(f"[CACHE] Loading existing CWE cache from {path}.")
                data = json.load(f)
                # Normalize JSON null -> Python None
                return {k: (v if v is not None else None) for k, v in data.items()}
        except json.JSONDecodeError:
            # Corrupted or partial cache file
            print(f"[CACHE ERROR] Could not decode cache file: {path}. Starting fresh.")
            return {}
    # No cache file found
    return {}


def save_cwe_cache(cache: Dict[str, str | None], path: str):
    """Saves the CWE cache dict to a JSON file."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[CACHE ERROR] Failed to save cache: {e}")


# =========================
# CWE HELPERS (simple + robust scraping)
# =========================

def fetch_cwe_description(cwe_id: str) -> str | None:
    """
    Fetches a short CWE description (first sentence) from the MITRE CWE page.

    Steps:
    1) Extract CWE number from cwe_id string (e.g., 'CWE-79' -> '79')
    2) Request the MITRE CWE definition page
    3) Locate the main description block (div#Description or fallback near h2 'Description')
    4) Remove noisy elements and return the first sentence
    """
    # Validate input type
    if not isinstance(cwe_id, str):
        return None

    # Extract CWE numeric id
    m = re.search(r"CWE-(\d+)", cwe_id)
    if not m:
        return None

    cwe_num = m.group(1)
    url = MITRE_CWE_URL.format(cwe_num)

    try:
        # Request the page with timeout to avoid hanging
        r = requests.get(url, headers=HEADERS, timeout=15)

        # Non-200 usually means blocked, missing page, or transient error
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        # Primary: use the official Description div
        desc_div = soup.find("div", {"id": "Description"})
        if not desc_div:
            # Fallback: find an h2 with "Description" text then take next sibling div
            h2 = soup.find("h2", string=re.compile("Description", re.IGNORECASE))
            if h2:
                desc_div = h2.find_next_sibling('div')
            if not desc_div:
                return None

        # Remove tags that often contain irrelevant content
        for junk_tag in desc_div(["script", "style", "h3", "h4", "table"]):
            junk_tag.decompose()

        # Extract clean text content
        text_content = desc_div.get_text(separator=' ', strip=True)

        if text_content:
            # Return only the first sentence for compactness
            first_sentence = text_content.split('.')[0].strip()
            return first_sentence + '.' if first_sentence else None

        return None

    except requests.exceptions.Timeout:
        # Network timeout (server slow or blocked)
        print(f"[CWE ERROR] Timeout fetching {cwe_id}.")
        return None
    except Exception as e:
        # Any other unexpected parsing/network error
        print(f"[CWE ERROR] Exception fetching {cwe_id}: {e}")
        return None


def add_cwe_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new 'cwe_description' column using MITRE CWE pages + caching.

    Pipeline:
    - Load cache
    - Collect unique CWE IDs in the dataset
    - Pre-fill 'CWE-Other' with a fixed description
    - Fetch missing CWEs sequentially with rate limiting
    - Save cache periodically and at the end
    - Map descriptions back into df['cwe_description']
    """
    # Ensure required column exists
    if "cwe_id" not in df.columns:
        raise ValueError("Missing 'cwe_id' column")

    # Load existing cache to avoid repeat calls
    cache = load_cwe_cache(CACHE_FILE)

    # Ensure stable comparison by converting to string and dropping NaNs
    unique_cwes = sorted(df["cwe_id"].astype(str).dropna().unique())

    # Special-case: 'CWE-Other' is not a standard numeric page
    CWE_OTHER_DESC = "Rare or uncommon weakness, grouped for statistical purposes."
    for cwe in ["CWE-Other", "CWE-OTHER"]:
        if cwe not in cache:
            cache[cwe] = CWE_OTHER_DESC
        if cwe in unique_cwes:
            unique_cwes.remove(cwe)

    # Only fetch CWEs not already in cache
    cwes_to_fetch = [cwe for cwe in unique_cwes if cwe not in cache]

    print(f"[CWE] Found {len(unique_cwes) + 1} total unique CWE IDs (including CWE-Other).")
    print(f"[CWE] {len(cache)} found/pre-filled in cache. Fetching {len(cwes_to_fetch)} new IDs.")

    # Fetch each missing CWE sequentially (polite to MITRE; easier to debug)
    for i, cwe in enumerate(cwes_to_fetch, start=1):
        # Defensive: if list is empty, skip
        if len(cwes_to_fetch) == 0:
            print("[CWE] All descriptions found in cache. Skipping fetch.")
            break

        print(f"[CWE] Fetching {cwe} ({i}/{len(cwes_to_fetch)})")

        description = fetch_cwe_description(cwe)
        cache[cwe] = description

        # Print fetched description (truncated)
        display_desc = description if description else "[ERROR/NONE] - Description not found."
        print(f"  -> DESC: {display_desc[:120]}...")

        # Save progress every N requests to prevent losing work
        if i % 50 == 0:
            save_cwe_cache(cache, CACHE_FILE)
            print("[CACHE] Intermediate save performed.")

        # Respect rate limit
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Final cache save
    save_cwe_cache(cache, CACHE_FILE)
    print("[CACHE] Final save performed.")

    # Map CWE IDs to descriptions
    df["cwe_description"] = df["cwe_id"].map(cache)
    return df


# =========================
# CVE + CODE FEATURES
# =========================

def enrich_cve_and_code_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features related to code and CVE metadata.

    Adds:
    - code_norm: stripped code (temporary)
    - code_key: stable hash of code_norm for dedup analysis
    - code_len: number of characters in code
    - code_lines: number of lines in code
    - cve_year / cve_number: parsed from cve_id
    - year_count: number of rows per cve_year
    - code_dup_count: number of rows sharing the same code_key
    - code_year_dup_count: number of rows sharing (code_key, cve_year)
    """
    out = df.copy()

    # Normalize code text for consistent hashing/dedup
    out["code_norm"] = out["code"].astype(str).str.strip()

    # Create a deterministic key for code duplicates analysis
    out["code_key"] = pd.util.hash_pandas_object(
        out["code_norm"], index=False
    ).astype("uint64")

    # Basic code-level features
    print("[FEATURES] Adding code length and line count")
    out["code_len"] = out["code_norm"].str.len()
    out["code_lines"] = out["code_norm"].str.count('\n') + 1

    # Parse CVE fields into numeric parts for analysis
    def parse_cve(x):
        # Missing or non-string CVE -> NaNs
        if not isinstance(x, str):
            return (np.nan, np.nan)
        m = CVE_RE.match(x.strip())
        if not m:
            return (np.nan, np.nan)
        return (int(m.group(1)), int(m.group(2)))

    parsed = out["cve_id"].apply(parse_cve)
    out["cve_year"] = parsed.apply(lambda t: t[0])
    out["cve_number"] = parsed.apply(lambda t: t[1])

    # Aggregation-based features
    out["year_count"] = out.groupby("cve_year")["cve_year"].transform("size")
    out["code_dup_count"] = out.groupby("code_key")["code_key"].transform("size")
    out["code_year_dup_count"] = (
        out.groupby(["code_key", "cve_year"])["code_key"].transform("size")
    )

    # Drop temporary normalized code column (keep code_key)
    out.drop(columns=["code_norm"], inplace=True, errors="ignore")

    return out


# =========================
# MAIN
# =========================

def main():
    # Force-reset cache file to ensure all CWEs are refetched from scratch
    if os.path.exists(CACHE_FILE):
        print(f"[FORCE RESET] Deleting old cache file: {CACHE_FILE}")
        os.remove(CACHE_FILE)

    print("[LOAD] Reading dataset...")
    df = pd.read_csv(INPUT_CSV)

    # Validate required columns are present for enrichment
    required = {"code", "cve_id", "cwe_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("[STEP 1] Add CWE descriptions")
    df = add_cwe_descriptions(df)

    print("[STEP 2] Add CVE year/number + code duplication features")
    df = enrich_cve_and_code_features(df)

    # Final cleanup: make cwe_description reliable for nunique/value_counts
    print("[CLEAN] Final cleanup for cwe_description column")
    df['cwe_description'] = df['cwe_description'].astype(str).str.strip()
    df.loc[df['cwe_description'].isin(['None', 'nan', '']), 'cwe_description'] = np.nan

    # Print summary of columns added in this enrichment stage
    print("\n[SUMMARY] New columns added:")
    added_cols = [
        "cwe_description",
        "cve_year",
        "cve_number",
        "code_len",
        "code_lines",
        "year_count",
        "code_dup_count",
        "code_year_dup_count",
        "code_key",
    ]
    for c in added_cols:
        print(" -", c)

    print(f"\n[SAVE] Writing to {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print("[DONE]")


if __name__ == "__main__":
    main()
