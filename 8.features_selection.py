import os
import numpy as np
import pandas as pd
import json
import gc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# =========================
# CONFIG
# =========================

# Root directory where embeddings, metadata, and outputs are stored
OUTPUT_DIR = "models/xgb_codebert_hybrid_ab_shap"

# Directory to save the filtered (feature-selected) train/test matrices
CLEAN_DATA_DIR = os.path.join(OUTPUT_DIR, "cleaned_dataset")

# Input paths (created by the embedding script)
EMBEDDING_PATH_TRAIN = os.path.join(OUTPUT_DIR, "X_train_emb.npy")
EMBEDDING_PATH_TEST = os.path.join(OUTPUT_DIR, "X_test_emb.npy")
META_PATH_TRAIN = os.path.join(OUTPUT_DIR, "df_train_meta.pkl")
META_PATH_TEST = os.path.join(OUTPUT_DIR, "df_test_meta.pkl")
LABELS_PATH_TRAIN = os.path.join(OUTPUT_DIR, "y_train.csv")
LABELS_PATH_TEST = os.path.join(OUTPUT_DIR, "y_test.csv")
NAMES_PATH = os.path.join(OUTPUT_DIR, "meta_feature_names.json")

# Voting threshold for feature selection:
# A feature is kept if it is selected by at least VOTE_THRESHOLD models
# NOTE: With VOTE_THRESHOLD=0, all features will pass (since Sum>=0 is always true)
VOTE_THRESHOLD = 0


def ensure_dir(path: str):
    """Create a directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_all_data():
    """
    Load:
    - y_train / y_test from CSV
    - X_train_emb / X_test_emb from memmap files on disk
    - meta features from pickled DataFrames
    - meta feature names from JSON

    Then:
    - concatenate embeddings + meta into a single numpy matrix per split
    - build the final list of feature names (emb_0..emb_767 + meta_names)
    """
    print("[LOAD] Reading all pre-computed files...")
    num_features = 768

    # Labels (flatten to 1D arrays)
    y_train = pd.read_csv(LABELS_PATH_TRAIN).values.ravel()
    y_test = pd.read_csv(LABELS_PATH_TEST).values.ravel()

    # Determine memmap shapes from label counts
    num_train, num_test = len(y_train), len(y_test)

    # Embeddings are stored as raw float32 arrays on disk (memmap)
    X_train_emb = np.memmap(
        EMBEDDING_PATH_TRAIN, dtype='float32', mode='r', shape=(num_train, num_features)
    )
    X_test_emb = np.memmap(
        EMBEDDING_PATH_TEST, dtype='float32', mode='r', shape=(num_test, num_features)
    )

    # Meta features (tabular) stored separately
    df_train_meta = pd.read_pickle(META_PATH_TRAIN).fillna(0)
    df_test_meta = pd.read_pickle(META_PATH_TEST).fillna(0)

    # Meta feature names for later mapping / export
    with open(NAMES_PATH, 'r') as f:
        meta_names = json.load(f)

    # Build unified feature name list for selection reports
    all_names = [f"emb_{i}" for i in range(num_features)] + meta_names

    # Concatenate embedding vectors + meta features into a single 2D array
    # np.array(memmap) forces loading into RAM; required for hstack + sklearn models
    X_train = np.nan_to_num(np.hstack([np.array(X_train_emb), df_train_meta.values]))
    X_test = np.nan_to_num(np.hstack([np.array(X_test_emb), df_test_meta.values]))

    # Free meta DataFrames from memory
    del df_train_meta, df_test_meta
    gc.collect()

    return X_train, X_test, y_train, y_test, all_names


def run_feature_voting(X_train, y_train, all_names):
    """
    Run a "feature voting" scheme using multiple models.

    Selection logic:
    - Linear models: a feature is selected if |coef| > 1e-5
    - Tree models: a feature is selected if importance > mean(importance)

    Output:
    - Full selection table CSV (per-model selection + Sum votes)
    - Winners table CSV for features with Sum >= VOTE_THRESHOLD
    """
    print(f"\n--- Phase 1: Feature Voting (Threshold > {VOTE_THRESHOLD - 1}) ---")

    # Standardize numeric scale for linear models (critical for L1 methods)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Initialize the selection table with feature names
    selection = pd.DataFrame({'Feature': all_names})

    # ---- Linear Models (L1-based sparsity) ----
    selection['Lasso'] = (np.abs(Lasso(alpha=0.01).fit(X_scaled, y_train).coef_) > 1e-5).astype(int)
    selection['ElasticNet'] = (np.abs(ElasticNet(alpha=0.01).fit(X_scaled, y_train).coef_) > 1e-5).astype(int)

    # Linear SVM with L1 penalty (sparse feature selection)
    selection['SVM'] = (
        np.abs(
            LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=2000)
            .fit(X_scaled, y_train)
            .coef_[0]
        ) > 1e-5
    ).astype(int)

    # Logistic regression with L1 penalty (sparse)
    selection['Logistic'] = (
        np.abs(
            LogisticRegression(penalty='l1', solver='liblinear')
            .fit(X_scaled, y_train)
            .coef_[0]
        ) > 1e-5
    ).astype(int)

    # Free standardized matrix to reduce RAM pressure
    del X_scaled
    gc.collect()

    # ---- Tree / Ensemble Models (feature_importances_) ----
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=10, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=100, verbose=-1),
        'AdaBoost': AdaBoostClassifier(n_estimators=50),
        'GradientBoost': GradientBoostingClassifier(n_estimators=50)
    }

    # Train each model and mark "selected" features by importance threshold
    for name, m in models.items():
        m.fit(X_train, y_train)
        importance = m.feature_importances_
        selection[name] = (importance > np.mean(importance)).astype(int)

    # Sum votes across all models
    selection['Sum'] = selection.drop('Feature', axis=1).sum(axis=1)

    # Sort by total votes (descending)
    selection = selection.sort_values('Sum', ascending=False).reset_index(drop=True)

    # Save full selection table (all features, all model votes)
    selection.to_csv(os.path.join(OUTPUT_DIR, "consensus_feature_selection_all_ccolumns.csv"), index=False)

    # Save only winners (features meeting threshold)
    winners_df = selection[selection['Sum'] >= VOTE_THRESHOLD]
    winners_df.to_csv(os.path.join(OUTPUT_DIR, "winners_features_min6_ccolumns.csv"), index=False)
    print(f"[INFO] Saved {len(winners_df)} winner features to winners_features_min6_ccolumns.csv")

    return selection


def save_cleaned_datasets(X_train, X_test, y_train, y_test, selection_df, all_names):
    """
    Create reduced train/test matrices using the selected feature set,
    and persist them for later modeling.
    """
    print(f"\n--- Phase 2: Cleaning & Saving Dataset (Threshold >= {VOTE_THRESHOLD}) ---")
    ensure_dir(CLEAN_DATA_DIR)

    # Get list of selected feature names
    final_var = selection_df[selection_df['Sum'] >= VOTE_THRESHOLD]['Feature'].tolist()

    # Convert feature names to column indices
    indices = [all_names.index(f) for f in final_var]

    # Slice matrices to keep only selected features
    X_train_clean = X_train[:, indices]
    X_test_clean = X_test[:, indices]

    # Persist reduced matrices
    np.save(os.path.join(CLEAN_DATA_DIR, "X_train_clean.npy"), X_train_clean)
    np.save(os.path.join(CLEAN_DATA_DIR, "X_test_clean.npy"), X_test_clean)

    # Persist selected feature names (to align later steps like SHAP)
    with open(os.path.join(CLEAN_DATA_DIR, "selected_features.json"), 'w') as f:
        json.dump(final_var, f)

    print(f"[INFO] New Dataset: {len(final_var)} features saved.")
    return X_train_clean, X_test_clean, final_var


def train_final_model(X_train_sub, X_test_sub, y_train, y_test):
    """
    Train a final XGBoost model using the feature-selected dataset.

    Uses scale_pos_weight to address class imbalance:
    ratio = (#negatives / #positives)
    """
    print(f"\n--- Phase 3: Final XGBoost Training ---")

    # Compute class imbalance ratio for XGBoost
    ratio = np.sum(y_train == 0) / np.sum(y_train == 1)

    # Train XGBoost with imbalance-aware weighting
    final_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=ratio,
        n_jobs=-1
    )
    final_model.fit(X_train_sub, y_train)

    # Save model to disk
    joblib.dump(final_model, os.path.join(OUTPUT_DIR, "final_xgb_model_ccolumns.joblib"))

    # Evaluate on test split
    y_pred = final_model.predict(X_test_sub)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # Load precomputed artifacts (embeddings + meta + labels)
    X_train, X_test, y_train, y_test, all_names = load_all_data()

    # Run multi-model voting for feature selection
    selection_df = run_feature_voting(X_train, y_train, all_names)

    # Save feature-selected datasets
    X_train_c, X_test_c, final_features = save_cleaned_datasets(
        X_train, X_test, y_train, y_test, selection_df, all_names
    )

    # Train final model using selected features
    train_final_model(X_train_c, X_test_c, y_train, y_test)






#
# ###########################################################
#
# Example run output:
#
# C:\Users\Amir\PycharmProjects\PythonProject3\.venv\Scripts\python.exe C:\Users\Amir\PycharmProjects\PythonProject3\8.features_selection.py
# [LOAD] Reading all pre-computed files...
#
# --- Phase 1: Feature Voting (Threshold > -1) ---
# [INFO] Saved 864 winner features to winners_features_min6_ccolumns.csv
#
# --- Phase 2: Cleaning & Saving Dataset (Threshold >= 0) ---
# [INFO] New Dataset: 864 features saved.
#
# --- Phase 3: Final XGBoost Training ---
#               precision    recall  f1-score   support
#
#            0       0.96      0.97      0.97      7343
#            1       0.29      0.27      0.28       364
#
#     accuracy                           0.93      7707
#    macro avg       0.62      0.62      0.62      7707
# weighted avg       0.93      0.93      0.93      7707
#
# Process finished with exit code 0
#
