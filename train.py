import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              precision_score, recall_score, f1_score,
                              confusion_matrix, precision_recall_curve)

df = pd.read_csv("data/creditcard.csv")

scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])

feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount_scaled", "Time_scaled"]
X = df[feature_cols]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}, fraud in test: {y_test.sum()}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, eval_metric="logloss", n_jobs=-1, random_state=42,
                              scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()),
}

results = []
probas = {}
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    dt = time.time() - t0
    probas[name] = proba
    pred = (proba >= 0.5).astype(int)
    results.append({
        "Model": name,
        "PR-AUC": average_precision_score(y_test, proba),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Precision@0.5": precision_score(y_test, pred),
        "Recall@0.5": recall_score(y_test, pred),
        "F1@0.5": f1_score(y_test, pred),
        "train_time_s": round(dt, 1),
    })
    print(f"{name} done in {dt:.1f}s")

results_df = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))

# Best model -> threshold exploration
best_name = results_df.iloc[0]["Model"]
best_proba = probas[best_name]
print(f"\nBest model: {best_name}")

for t in [0.3, 0.5, 0.7, 0.9]:
    pred = (best_proba >= t).astype(int)
    p, r, f1 = precision_score(y_test, pred), recall_score(y_test, pred), f1_score(y_test, pred)
    print(f"threshold={t}: precision={p:.3f} recall={r:.3f} f1={f1:.3f}")

# Feature importance
if best_name in ("Random Forest", "XGBoost"):
    importances = pd.Series(models[best_name].feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n=== {best_name} Feature Importance (top 10) ===")
    print(importances.head(10).to_string())

results_df.to_csv("model_comparison_results.csv", index=False)
print("\nSaved model_comparison_results.csv")
