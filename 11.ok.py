import os
import numpy as np
import pandas as pd
import json
import joblib
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# ==========================================
# 1) PATH CONFIG + DATA LOADING
# ==========================================

# Absolute path of the current script (useful for debugging relative paths)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Root artifacts directory from previous pipeline steps
OUTPUT_DIR = "models/xgb_codebert_hybrid_ab_shap"

# Directory containing feature-selected train matrix and selected feature names
CLEAN_DATA_DIR = os.path.join(OUTPUT_DIR, "cleaned_dataset")

# Output directory for final balanced models / ensembles
BALANCED_RES_DIR = os.path.join(OUTPUT_DIR, "dataset_balanced_results")
os.makedirs(BALANCED_RES_DIR, exist_ok=True)


def load_data():
    """
    Loads the feature-selected training matrix and labels.

    Returns:
    - X: float32 numpy array (N, d)
    - y: 1D labels array
    """
    X = np.load(os.path.join(CLEAN_DATA_DIR, "X_train_clean.npy"), allow_pickle=True).astype('float32')
    y = pd.read_csv(os.path.join(OUTPUT_DIR, "y_train.csv")).values.ravel()
    return X, y


# Load full training data (before dev/test split)
X_full, y_full = load_data()

# Split is required before tuning to prevent leakage and to measure generalization
from sklearn.model_selection import train_test_split

# 70% train, 15% dev, 15% test
X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
    X_full, y_full, test_size=0.30, random_state=222
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=222
)

# Normalize features (especially important for some models and for consistent scale)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_raw)
X_dev_s = scaler.transform(X_dev)
X_test_s = scaler.transform(X_test)

# Oversample training data only to handle class imbalance
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train_s, y_train_raw)


# ==========================================
# 2) EVALUATION HELPER
# ==========================================

def evaluate_improvement(base_model, tuned_model, X_val, y_val, model_name):
    """
    Compare base vs tuned models on the validation set using F1-score.
    Prints absolute performance and relative improvement percentage.
    """
    base_f1 = f1_score(y_val, base_model.predict(X_val), zero_division=0)
    tuned_f1 = f1_score(y_val, tuned_model.predict(X_val), zero_division=0)

    print(f"\n--- {model_name} Analysis ---")
    print(f"Base F1: {base_f1:.4f} | Tuned F1: {tuned_f1:.4f}")

    if base_f1 > 0:
        imp = 100 * (tuned_f1 - base_f1) / base_f1
        print(f"Improvement: {imp:.2f}%")

    return tuned_f1


# ==========================================
# 3) PART 1: DEEPER TUNING
# ==========================================

# --- CatBoost tuning ---
print("[RUN] Training CatBoost...")

# Base model fit (baseline comparison)
base_cat = CatBoostClassifier(verbose=0, random_state=222).fit(X_train, y_train)

# Random search over key CatBoost hyperparameters
cat_search = RandomizedSearchCV(
    CatBoostClassifier(verbose=0, random_state=222),
    param_distributions={
        'depth': [4, 6, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [500, 1000]
    },
    n_iter=5,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=222
)

# Fit random search on balanced training set
cat_search.fit(X_train, y_train)

# Best tuned CatBoost model
best_cat = cat_search.best_estimator_

# Evaluate tuned vs base on DEV
evaluate_improvement(base_cat, best_cat, X_dev_s, y_dev, "CatBoost")

# --- RandomForest tuning ---
print("\n[RUN] Training RandomForest...")

# Base RF fit (baseline)
base_rf = RandomForestClassifier(random_state=222).fit(X_train, y_train)

# Random search over common RF knobs
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=222),
    param_distributions={
        'n_estimators': [200, 500],
        'max_depth': [10, 20, None]
    },
    n_iter=5,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=222
)

# Fit random search
rf_search.fit(X_train, y_train)

# Best tuned RF model
best_rf = rf_search.best_estimator_

# Evaluate tuned vs base on DEV
evaluate_improvement(base_rf, best_rf, X_dev_s, y_dev, "RandomForest")


# ==========================================
# 4) PART 2: WEIGHTED SOFT-VOTING ENSEMBLE
# ==========================================

def find_best_weights(model_a, model_b, X_val, y_val):
    """
    Grid-search integer weights for a 2-model soft voting ensemble.

    Logic:
    - Get class probabilities from each model
    - Weighted-average probabilities
    - Convert to predicted class with argmax
    - Choose weights that maximize F1 on validation
    """
    best_w = (1, 1)
    max_f1 = 0

    prob_a = model_a.predict_proba(X_val)
    prob_b = model_b.predict_proba(X_val)

    for w_a in range(1, 5):
        for w_b in range(1, 5):
            avg_prob = (prob_a * w_a + prob_b * w_b) / (w_a + w_b)
            f1 = f1_score(y_val, np.argmax(avg_prob, axis=1), zero_division=0)

            if f1 > max_f1:
                max_f1 = f1
                best_w = (w_a, w_b)

    return best_w


# Find best ensemble weights on DEV
opt_w = find_best_weights(best_cat, best_rf, X_dev_s, y_dev)
print(f"\n[ENSEMBLE] Optimal Weights: CatBoost={opt_w[0]}, RF={opt_w[1]}")

# Train final soft-voting ensemble on balanced training set
final_model = VotingClassifier(
    estimators=[('cat', best_cat), ('rf', best_rf)],
    voting='soft',
    weights=[opt_w[0], opt_w[1]]
).fit(X_train, y_train)

# Final evaluation on TEST split (untouched)
print("\n=== FINAL TEST REPORT ===")
print(classification_report(y_test, final_model.predict(X_test_s)))

# Save final ensemble model
joblib.dump(final_model, os.path.join(BALANCED_RES_DIR, "final_weighted_ensemble.joblib"))


# ########################################################
# Example run output (kept as comments):
#
# C:\Users\Amir\PycharmProjects\PythonProject3\.venv\Scripts\python.exe C:\Users\Amir\PycharmProjects\PythonProject3\11.ok.py
# [RUN] Training CatBoost...
#
# --- CatBoost Analysis ---
# Base F1: 0.2750 | Tuned F1: 0.2730
# Improvement: -0.73%
#
# [RUN] Training RandomForest...
#
# --- RandomForest Analysis ---
# Base F1: 0.2050 | Tuned F1: 0.2055
# Improvement: 0.23%
#
# [ENSEMBLE] Optimal Weights: CatBoost=2, RF=1
#
# === FINAL TEST REPORT ===
#               precision    recall  f1-score   support
#
#            0       0.96      0.98      0.97      6017
#            1       0.34      0.21      0.26       327
#
# Process finished with exit code 0
#
