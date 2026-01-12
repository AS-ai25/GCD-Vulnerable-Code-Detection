import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import json
import joblib  # ספרייה לשמירת מודלים
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# =========================
# 1. הגדרות נתיבים
# =========================
OUTPUT_DIR = "models/xgb_codebert_hybrid_ab_shap"
CLEAN_DATA_DIR = os.path.join(OUTPUT_DIR, "cleaned_dataset")
BALANCED_RES_DIR = os.path.join(OUTPUT_DIR, "dataset_balanced_results")

os.makedirs(BALANCED_RES_DIR, exist_ok=True)


def load_clean_data():
    print("[LOAD] Reading CLEANED dataset (Features strictly > 6 votes)...")


    X_train = np.load(os.path.join(CLEAN_DATA_DIR, "X_train_clean.npy"), allow_pickle=True).astype('float32')
    X_test = np.load(os.path.join(CLEAN_DATA_DIR, "X_test_clean.npy"), allow_pickle=True).astype('float32')

    y_train = pd.read_csv(os.path.join(OUTPUT_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(OUTPUT_DIR, "y_test.csv")).values.ravel()

    with open(os.path.join(CLEAN_DATA_DIR, "selected_features.json"), 'r') as f:
        feature_names = json.load(f)

    print(f"[INFO] Dataset Loaded. Features count: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, feature_names


X_train, X_test, y_train, y_test, feature_names = load_clean_data()

# =========================
# 2. השוואת אסטרטגיות איזון
# =========================
strategies = {
    "XGB Simple (Weighted)": None,
    "XGB + RUS": RandomUnderSampler(random_state=42),
    "XGB + ROS": RandomOverSampler(random_state=42),
    "XGB + SMOTE": SMOTE(random_state=42),
    "XGB + SMOTETomek": SMOTETomek(random_state=42)
}

results = []
trained_models = {}

for name, sampler in strategies.items():
    print(f"[RUN] Processing {name}...")
    if sampler is None:
        X_res, y_res = X_train, y_train
        ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
        model = XGBClassifier(random_state=42, scale_pos_weight=ratio, eval_metric='logloss')
    else:
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        model = XGBClassifier(random_state=42, eval_metric='logloss')

    model.fit(X_res, y_res)
    trained_models[name] = model
    y_pred = model.predict(X_test)

    results.append({
        "Method": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0)
    })
    if sampler is not None: del X_res, y_res; gc.collect()

# =========================
# 3. שמירת תוצאות וגרף השוואה
# =========================
results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
results_df.to_csv(os.path.join(BALANCED_RES_DIR, "balancing_comparison_metrics.csv"), index=False)

plt.figure(figsize=(10, 6))
results_df_plot = results_df.set_index('Method')[['Precision', 'Recall', 'F1-Score']]
results_df_plot.plot(kind='bar', ax=plt.gca())
plt.title('Performance Comparison (Cleaned Features > 6)')
plt.ylabel('Score')
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(BALANCED_RES_DIR, "comparison_chart.png"), dpi=300)
print(results_df.to_string(index=False))

# ==========================================================
# 4. ניתוח חשיבות ושמירת המודל המנצח (XGB + ROS)
# ==========================================================
print("\n" + "=" * 50)
print("PHASE 4: SAVING MODEL & FEATURE IMPORTANCE")
print("=" * 50)

best_model_name = "XGB + ROS"
if best_model_name in trained_models:
    best_model = trained_models[best_model_name]

    # --- שמירת המודל ---
    model_save_path = os.path.join(BALANCED_RES_DIR, "best_xgb_ros_model.joblib")
    joblib.dump(best_model, model_save_path)
    print(f"[SAVE] Best model saved to: {model_save_path}")

    # --- ניתוח חשיבות פיצ'רים ---
    importances = best_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='Importance', y='Feature', data=feat_imp_df,
        hue='Feature', palette='viridis', legend=False
    )
    plt.title(f'Feature Importance: {best_model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(BALANCED_RES_DIR, "feature_importance_top17.png"), dpi=300)

print("\n[SUCCESS] Process completed. Model and results are ready.")
plt.show()