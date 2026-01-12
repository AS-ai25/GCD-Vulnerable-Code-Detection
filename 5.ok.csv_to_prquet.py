# This code assumes execution in an environment with sufficient RAM
# It converts a large CSV dataset into a Parquet file for faster I/O and better compression

import pandas as pd

# Input CSV file containing the enriched vulnerability dataset
FILE_NAME = "ok.clean_vuln_dataset_enriched.csv"

# Output Parquet file (columnar, compressed, analytics-friendly format)
PARQUET_NAME = "ok.clean_vuln_dataset_full.parquet"

# Load the entire CSV into memory
# low_memory=False forces pandas to read the file in one pass
# and infer dtypes more consistently (uses more RAM)
df = pd.read_csv(FILE_NAME, low_memory=False)

# Save the DataFrame to Parquet format without the index column
# Parquet is preferred for large-scale analytics and ML pipelines
df.to_parquet(PARQUET_NAME, index=False)
