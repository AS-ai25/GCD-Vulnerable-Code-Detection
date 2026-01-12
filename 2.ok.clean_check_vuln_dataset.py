import pandas as pd

# Load the combined vulnerability dataset from CSV
df = pd.read_csv("combined_vuln_dataset_with_meta.csv")

# Remove rows where the 'code' column is NaN (missing source code)
df = df.dropna(subset=["code"])

# Ensure 'code' is string type and remove rows with empty or whitespace-only code
df["code"] = df["code"].astype(str)
df = df[df["code"].str.strip() != ""]

# Threshold definition:
# Columns with more than 75% missing values will be dropped
THRESHOLD = 0.25

# Compute the fraction of missing (NaN) values per column
null_ratio = df.isna().mean()

# Identify columns exceeding the missing-value threshold
cols_to_drop = null_ratio[null_ratio > THRESHOLD].index.tolist()

# Log columns that will be removed and their missing percentages
print("Dropping columns (null % > 75%):")
for c in cols_to_drop:
    print(f"{c}: {null_ratio[c]*100:.2f}%")

# Drop the selected high-null columns
df = df.drop(columns=cols_to_drop)

# Display remaining column names after cleanup
print("\nRemaining columns:")
print(df.columns.tolist())

# Print DataFrame schema and basic diagnostics
print(df.info())
print(df.shape)
print(df.head())

# Optional inspection helpers (kept commented out)
# print(df[["dataset", "cve_id", "cwe_id"]].head())
# print(df["cve_id"].value_counts(dropna=False).head())
# print(df["cwe_id"].value_counts(dropna=False).head())

# Build a detailed null-value summary table
null_summary = pd.DataFrame({
    "null_count": df.isna().sum(),
    "null_percent": (df.isna().sum() / len(df)) * 100
}).sort_values("null_count", ascending=False)

# Print null-value summary for remaining columns
print(null_summary)

# Row count before any optional row-wise dropna operations
print("Rows before dropna:", len(df))

# Optional row-level cleaning strategies (disabled by default):
# - Drop rows with any missing value
# - Drop rows missing CWE information
# df = df.dropna(axis=0, how="any")
# df = df.dropna(subset=['cwe_id'])

# Row count after optional dropna (currently unchanged)
print("Rows after dropna:", len(df))

###################################
# Output path for the cleaned dataset
OUT_CSV = "ok.clean_vuln_dataset.csv"

# Save the cleaned DataFrame to CSV
df.to_csv(OUT_CSV, index=False)

# Confirm successful save
print(f"Saved clean dataset to: {OUT_CSV}")
#######################33333
