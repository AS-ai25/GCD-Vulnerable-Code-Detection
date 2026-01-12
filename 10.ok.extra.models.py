import os
import numpy as np
import pandas as pd
import json
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

# Advanced models
import xgboost
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# =========================
# 1) PATH CONFIG
# =========================

# Absolute directory of this script file (useful for relative path debugging)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Root output folder containing artifacts from previous steps
OUTPUT_DIR = "models/xgb_codebert_hybrid_ab_shap"

# Directory with feature-selected matrices and feature list
CLEAN_DATA_DIR = os.path.join(OUTPUT_DIR, "cleaned_dataset")

# Output directory for balanced training results + final artifacts
BALANCED_RES_DIR = os.path.join(OUTPUT_DIR, "dataset_balanced_results")

# Ensure output directory exists
os.makedirs(BALANCED_RES_DIR, exist_ok=True)


def load_final_clean_data():
    """
    Load the feature-selected training matrix and labels.

    Returns:
    - X: float32 numpy array (feature-selected training matrix)
    - y: 1D labels array
    - feature_names: list of selected feature names aligned with X columns
    """
    X = np.load(os.path.join(CLEAN_DATA_DIR, "X_train_clean.npy"), allow_pickle=True).astype('float32')
    y = pd.read_csv(os.path.join(OUTPUT_DIR, "y_train.csv")).values.ravel()
    with open(os.path.join(CLEAN_DATA_DIR, "selected_features.json"), 'r') as f:
        feature_names = json.load(f)
    return X, y, feature_names


# Load the final clean dataset
X_full, y_full, feature_names = load_final_clean_data()

# =========================
# 2) SPLIT + SCALE + BALANCE
# =========================

# Split into Train / Dev / Test:
# - Train: 70%
# - Dev:   15%
# - Test:  15%
X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
    X_full, y_full, test_size=0.30, random_state=222
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=222
)

# Standard scaling (important for linear models; harmless for tree models)
scaler = StandardScaler()
X_train_raw = scaler.fit_transform(X_train_raw)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# Random oversampling on training data only (balances class distribution)
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train_raw, y_train_raw)

print(f"[INFO] Data Balanced. Training shape: {X_train.shape}")

# =========================
# 3) BASELINE MODEL COMPARISON
# =========================

# Binary check (kept for safety / future extension)
is_binary = len(np.unique(y_train)) == 2

# XGBoost config for binary classification
xgb_kwargs = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42
}

# Candidate models (mix of linear, bagging, boosting, and gradient methods)
models_dict = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": xgboost.XGBClassifier(**xgb_kwargs),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
}

results = []
trained_objects = {}

# Train and evaluate on DEV split to pick the best family before tuning
for name, model in models_dict.items():
    print(f"[RUN] Training: {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_dev)

    f1 = metrics.f1_score(y_dev, y_pred, zero_division=0)
    prec = metrics.precision_score(y_dev, y_pred, zero_division=0)
    rec = metrics.recall_score(y_dev, y_pred, zero_division=0)

    results.append({"Model": name, "F1": f1, "Precision": prec, "Recall": rec})
    trained_objects[name] = model

df_results = pd.DataFrame(results).sort_values(by="F1", ascending=False)
print("\n=== BASELINE COMPARISON (ALL MODELS) ===")
print(df_results)

# =========================
# 4) AUTOMATED HYPERPARAMETER TUNING
# =========================

# Pick best model by DEV F1-score
best_model_name = df_results.iloc[0]['Model']
print(f"\n[*] Tuning the winner: {best_model_name}")

# Define different search spaces depending on the winner type
if any(m in best_model_name for m in ["XGB", "LightGBM", "CatBoost", "Gradient"]):
    param_dist = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 9],
        'subsample': [0.8, 1.0]
    }
elif "Forest" in best_model_name or "Extra" in best_model_name:
    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
else:
    # Logistic Regression
    param_dist = {'C': [0.1, 1, 10]}

# RandomizedSearchCV:
# - uses 3-fold CV on the (oversampled) training set
# - optimizes F1 score
random_search = RandomizedSearchCV(
    estimator=trained_objects[best_model_name],
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
best_tuned_model = random_search.best_estimator_

# =========================
# 5) FINAL EVALUATION + SHAP
# =========================

# Evaluate on untouched TEST split
y_test_pred = best_tuned_model.predict(X_test)
print("\n=== FINAL TEST PERFORMANCE (TUNED) ===")
print(metrics.classification_report(y_test, y_test_pred))

print("[SHAP] Generating Explainability Plots...")

# Choose a SHAP explainer suitable for the model type
if "Logistic" in best_model_name:
    explainer = shap.LinearExplainer(best_tuned_model, X_train)
else:
    explainer = shap.TreeExplainer(best_tuned_model)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Handle SHAP output shape differences:
# - some explainers return a list [class0, class1] for binary classification
if isinstance(shap_values, list) and len(shap_values) > 1:
    shap_val_plot = shap_values[1]
else:
    shap_val_plot = shap_values

# Save SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_val_plot, X_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(BALANCED_RES_DIR, "final_shap_analysis.png"))
plt.show()

# Save artifacts: model, scaler, and feature list
joblib.dump(best_tuned_model, os.path.join(BALANCED_RES_DIR, "final_best_model.joblib"))
joblib.dump(scaler, os.path.join(BALANCED_RES_DIR, "final_scaler.joblib"))
with open(os.path.join(BALANCED_RES_DIR, "final_features.json"), 'w') as f:
    json.dump(feature_names, f)

print(f"\n[SUCCESS] Best Model: {best_model_name} | Artifacts saved in {BALANCED_RES_DIR}")


# ##################################################
# Example run output (kept as comments):
#
# C:\Users\Amir\PycharmProjects\PythonProject3\.venv\Scripts\python.exe C:\Users\Amir\PycharmProjects\PythonProject3\10.ok.extra.models.py
# [INFO] Data Balanced. Training shape: (56078, 17)
# [RUN] Training: LogisticRegression...
# [RUN] Training: RandomForest...
# [RUN] Training: ExtraTrees...
# [RUN] Training: GradientBoosting...
# [RUN] Training: XGBoost...
# [RUN] Training: LightGBM...
# UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
# [RUN] Training: CatBoost...
#
# === BASELINE COMPARISON (ALL MODELS) ===
#                 Model        F1  Precision    Recall
# 6            CatBoost  0.254879   0.211832  0.319885
# 4             XGBoost  0.236842   0.191150  0.311239
# 5            LightGBM  0.231939   0.148660  0.527378
# 3    GradientBoosting  0.208875   0.123673  0.671470
# 1        RandomForest  0.200456   0.478261  0.126801
# 0  LogisticRegression  0.178109   0.103088  0.654179
# 2          ExtraTrees  0.097436   0.441860  0.054755
#
# [*] Tuning the winner: CatBoost
# Fitting 3 folds for each of 10 candidates, totalling 30 fits
#
# === FINAL TEST PERFORMANCE (TUNED) ===
#               precision    recall  f1-score   support
#
#            0       0.96      0.98      0.97      6017
#            1       0.35      0.17      0.23       327
#
# [SHAP] Generating Explainability Plots...
#
# [SUCCESS] Best Model: CatBoost | Artifacts saved in models/xgb_codebert_hybrid_ab_shap\dataset_balanced_results
#
# Process finished with exit code 0
#
