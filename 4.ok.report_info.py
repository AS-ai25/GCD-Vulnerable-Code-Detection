import pandas as pd
import os
import numpy as np


# ================= CONFIG =================
# FILE_NAME = "ok.clean_vuln_dataset_enriched.csv"
FILE_NAME = "ok.clean_vuln_dataset_imputed.csv"


# ================= CODE =================

try:
    # 1) Load the DataFrame from CSV
    df = pd.read_csv(FILE_NAME)
    print(f"✅ Successfully loaded file: '{FILE_NAME}'.")
    print("-" * 70)

    # 2) Print df.info() summary (schema, dtypes, non-null counts, memory usage)
    print("\n## 📋 DataFrame structure (df.info())")
    df.info()
    print("-" * 70)

    # 3) Print full list of columns
    print("\n## 📌 All columns:")
    print(list(df.columns))
    print("-" * 70)

    # 4) Build summary report: dtype, unique count, null count, null percentage
    print("\n## 📊 Summary report - uniqueness and missing values")

    total_rows = len(df)

    # Compute per-column stats
    summary_df = pd.DataFrame({
        'Dtype': df.dtypes,                 # column data types
        'Unique Count': df.nunique(),        # number of distinct values per column
        'Null Count': df.isnull().sum(),     # number of missing values per column
    })

    # Add missing percentage (formatted as string with %)
    summary_df['Null Percentage'] = (summary_df['Null Count'] / total_rows * 100).round(2).astype(str) + '%'

    # Reorder columns for readability
    summary_df = summary_df[['Dtype', 'Unique Count', 'Null Count', 'Null Percentage']]

    # Print full summary table
    print(summary_df)
    print("-" * 70)

    # Highlight only columns with missing values (actionable subset)
    null_report = summary_df[summary_df['Null Count'] > 0]
    if not null_report.empty:
        print("\n🛑 Columns with missing values (need handling):")
        print(null_report)
    else:
        print("\n🎉 No missing (Null) values in the DataFrame.")

except FileNotFoundError:
    # File not found: show current working directory to help user locate file
    print(f"\n[ERROR] File '{FILE_NAME}' was not found in: {os.getcwd()}")
    print("Please verify the file name and path.")
except Exception as e:
    # Any other exception during load/processing
    print(f"\n[ERROR] An error occurred while loading the file: {e}")

# ----------------------------------------------------------------------
# Extra section (from 4.ok.report_info.py) - after df is loaded
# ----------------------------------------------------------------------

# Validate and recalculate uniqueness/nulls for cwe_description after explicit cleaning
if 'cwe_description' in df.columns:
    print(f"\n### 🔍 Direct cleaned check for cwe_description (unique count) ###")

    # 1) Convert to string, strip whitespace, and normalize placeholders to real NaN
    cleaned_descriptions = (
        df['cwe_description'].astype(str)
        .str.strip()
        .replace({'None': np.nan, 'nan': np.nan, '': np.nan})
    )

    # 2) Unique count after cleaning (excluding NaNs)
    unique_count_cleaned = cleaned_descriptions.nunique(dropna=True)

    # 3) Missing count + missing percentage after cleaning
    null_count_cleaned = cleaned_descriptions.isnull().sum()
    null_percentage_cleaned = round((null_count_cleaned / len(df)) * 100, 2)

    print("** After direct cleaning in report:**")
    print(f" - Unique Count: {unique_count_cleaned}")
    print(f" - Null Count: {null_count_cleaned} ({null_percentage_cleaned}%)")
    print("----------------------------------------------------------------------")

# ... (the rest of the reporting code remains unchanged) ...
