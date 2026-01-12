# GCD-Vulnerable-Code-Detection
An End-to-End Pipeline for Explainable Vulnerable Code Detection


# Vulnerability Dataset Pipeline (MegaVul + BigVul + DiverseVul + Juliet) + CodeBERT + ML

End-to-end pipeline for:
1) Building a unified vulnerability dataset from public sources (HuggingFace + local Juliet)
2) Cleaning and enriching the dataset (CWE descriptions, CVE parsing, code features)
3) Converting to Parquet for efficiency
4) Imputation (IDs + numeric)
5) Generating CodeBERT embeddings (incremental disk saving)
6) Feature voting + training (XGBoost and other models)
7) Class-imbalance strategies comparison
8) Advanced model search + SHAP explainability
9) Optional tuned weighted ensemble (CatBoost + RandomForest)

---

## Project Structure (11 Files)

> Names below are the recommended filenames matching the scripts you posted.

1. `1.build_combined_vuln_dataset.py`  
   Build a combined dataset from:
   - HuggingFace: `hitoshura25/megavul`
   - HuggingFace: `bstee615/bigvul`
   - HuggingFace: `bstee615/diversevul` (best-effort; continues if fails)
   - Local: Juliet test suite (`data/juliet`)  
   Output:
   - `combined_vuln_dataset_with_meta.csv`
   - `combined_vuln_dataset_with_meta.parquet` (if parquet engine installed)

2. `2.clean_drop_sparse_columns.py`  
   Loads combined CSV, drops columns with too many missing values, saves:
   - `ok.clean_vuln_dataset.csv`

3. `3.enrich_cwe_and_features.py`  
   Enriches dataset by:
   - Fetching CWE descriptions from MITRE (with JSON caching)
   - Adding CVE parsing + code features (len, lines, dup counts, hash key)  
   Output:
   - `ok.clean_vuln_dataset_enriched.csv`

4. `4.report_info.py`  
   Reporting utility:
   - `df.info()`, column list
   - Unique/null report per column
   - Additional cleaned unique/null check for `cwe_description`

5. `5.csv_to_parquet_full.py`  
   Converts the enriched CSV into a full Parquet file:
   - `ok.clean_vuln_dataset_full.parquet`

6. `6.impute_dataset_from_parquet.py`  
   Reads Parquet (only required columns), imputes:
   - ID columns: `cve_id`, `cwe_id`
   - Numeric columns: fills NaN with `-1.0`
   Outputs:
   - `ok.clean_vuln_dataset_imputed.csv`
   - `ok.clean_vuln_dataset_imputed.parquet`

7. `7.embed_codebert_to_disk.py`  
   Generates CodeBERT embeddings for train/test code splits:
   - Uses `GroupShuffleSplit` by `cve_id` to reduce leakage
   - Saves embeddings incrementally using `np.memmap` to avoid RAM crashes
   Outputs (under `models/xgb_codebert_hybrid_ab_shap/`):
   - `X_train_emb.npy`, `X_test_emb.npy` (memmap files)
   - `df_train_meta.pkl`, `df_test_meta.pkl`
   - `y_train.csv`, `y_test.csv`
   - `meta_feature_names.json`

8. `8.feature_voting_and_xgb_train.py`  
   Feature voting using multiple models (linear + tree-based), then:
   - Saves full selection table
   - Saves winner features CSV
   - Saves reduced train/test matrices
   - Trains final XGBoost
   Outputs:
   - `consensus_feature_selection_all_ccolumns.csv`
   - `winners_features_min6_ccolumns.csv`
   - `cleaned_dataset/X_train_clean.npy`, `cleaned_dataset/X_test_clean.npy`
   - `cleaned_dataset/selected_features.json`
   - `final_xgb_model_ccolumns.joblib`

9. `9.balancing_strategies_compare.py`  
   Compares imbalance strategies:
   - Weighted XGB
   - RUS / ROS / SMOTE / SMOTETomek
   Saves PR curves + confusion matrices + metrics CSV + best model:
   - `dataset_balanced_results_all/pr_curve_comparison.png`
   - `dataset_balanced_results_all/all_confusion_matrices.png`
   - `dataset_balanced_results_all/balancing_comparison_metrics.csv`
   - `dataset_balanced_results_all/best_model.joblib`

10. `10.extra_models_tuning_shap.py`  
   Trains multiple model families on balanced training set, picks best by DEV F1,
   does `RandomizedSearchCV`, evaluates on TEST, then SHAP summary plot:
   Outputs:
   - `dataset_balanced_results/final_best_model.joblib`
   - `dataset_balanced_results/final_scaler.joblib`
   - `dataset_balanced_results/final_features.json`
   - `dataset_balanced_results/final_shap_analysis.png`

11. `11.weighted_ensemble_cat_rf.py`  
   Deep tuning for CatBoost + RandomForest + soft voting ensemble with optimized weights on DEV.
   Output:
   - `dataset_balanced_results/final_weighted_ensemble.joblib`

---

## Recommended Run Order

1) Build combined dataset:
```bash
python 1.build_combined_vuln_dataset.py

