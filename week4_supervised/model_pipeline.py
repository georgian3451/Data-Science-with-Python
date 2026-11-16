import os
import sys
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # run from anywhere; paths below are relative to this folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay
)

from src.preprocessing import (
    TitanicFeaturePipeline,
    get_feature_column_transformer,
    build_titanic_pipeline,
)

sns.set_style("whitegrid")
np.random.seed(42)

# ============================================================
# 1. LOAD RAW DATA & SPLIT (Leakage-Safe Architecture)
# ============================================================
# Critical improvement: we load raw data and split into train/test FIRST.
# All imputation medians, IQR capping bounds, one-hot encodings, scaling,
# and interaction baselines are fit strictly on training data / inside CV folds.
raw = pd.read_csv("../data/raw/titanic.csv")
print("Raw dataset shape:", raw.shape)
print("Class balance:\n", raw["Survived"].value_counts(normalize=True).round(3))

X = raw.drop(columns=["Survived"])
y = raw["Survived"]

# Stratified 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print("Train class balance:\n", y_train.value_counts(normalize=True).round(3))
print("Test class balance:\n", y_test.value_counts(normalize=True).round(3))

# ============================================================
# 2. MODEL COMPARISON VIA 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=7), True),
    "Decision Tree": (DecisionTreeClassifier(max_depth=5, random_state=42), False),
    "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=42), False),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), False),
    "SVM (RBF kernel)": (SVC(probability=True, random_state=42), True),
}

scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_results = []

for name, (model, needs_scaling) in models.items():
    pipe = build_titanic_pipeline(model, needs_scaling=needs_scaling, include_interactions=True)
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    cv_results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean(),
        "ROC_AUC": scores["test_roc_auc"].mean(),
        "Accuracy_std": scores["test_accuracy"].std(),
    })

cv_df = pd.DataFrame(cv_results).sort_values("F1", ascending=False).reset_index(drop=True)
print("\n===== 5-FOLD CV MODEL COMPARISON (Leakage-Safe on Training Set) =====")
print(cv_df.round(4))
cv_df.to_csv("results/model_comparison_cv.csv", index=False)

# Bar chart of CV comparison
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(cv_df))
width = 0.2
ax.bar(x - 1.5*width, cv_df["Accuracy"], width, label="Accuracy", color="#4C72B0")
ax.bar(x - 0.5*width, cv_df["F1"], width, label="F1", color="#55A868")
ax.bar(x + 0.5*width, cv_df["ROC_AUC"], width, label="ROC-AUC", color="#C44E52")
ax.bar(x + 1.5*width, cv_df["Recall"], width, label="Recall", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(cv_df["Model"], rotation=20, ha="right")
ax.set_ylabel("Score (5-fold CV mean)")
ax.set_title("Model Comparison — 5-Fold Cross-Validation on Training Set (Leakage-Safe)")
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("figures/fig1_model_comparison.png", dpi=140)
plt.close()

# ============================================================
# 3. HYPERPARAMETER TUNING (Random Forest & Gradient Boosting)
# ============================================================
best_model_name = cv_df.iloc[0]["Model"]
print(f"\nTop baseline model by CV F1: {best_model_name}")

pipe_rf = build_titanic_pipeline(
    RandomForestClassifier(random_state=42), needs_scaling=False, include_interactions=True
)
param_rf = {
    "clf__n_estimators": [200, 300, 500],
    "clf__max_depth": [4, 6, 8, None],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2"],
}
grid_rf = GridSearchCV(pipe_rf, param_rf, cv=cv, scoring="f1", n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("\nBest hyperparameters (Random Forest):", grid_rf.best_params_)
print("Best CV F1 score (Random Forest):", round(grid_rf.best_score_, 4))

pipe_gb = build_titanic_pipeline(
    GradientBoostingClassifier(random_state=42), needs_scaling=False, include_interactions=True
)
param_gb = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [2, 3, 4],
    "clf__learning_rate": [0.05, 0.1],
}
grid_gb = GridSearchCV(pipe_gb, param_gb, cv=cv, scoring="f1", n_jobs=-1)
grid_gb.fit(X_train, y_train)
print("\nBest hyperparameters (Gradient Boosting):", grid_gb.best_params_)
print("Best CV F1 score (Gradient Boosting):", round(grid_gb.best_score_, 4))

final_pipeline = grid_rf.best_estimator_ if grid_rf.best_score_ >= grid_gb.best_score_ else grid_gb.best_estimator_
final_model_name = "Random Forest (tuned)" if final_pipeline is grid_rf.best_estimator_ else "Gradient Boosting (tuned)"
print(f"\nFinal chosen model: {final_model_name}")

# ============================================================
# 4. FINAL EVALUATION ON HELD-OUT TEST SET
# ============================================================
final_pipeline.fit(X_train, y_train)
y_pred = final_pipeline.predict(X_test)
y_proba = final_pipeline.predict_proba(X_test)[:, 1]

test_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "ROC_AUC": roc_auc_score(y_test, y_proba),
}
print("\n===== FINAL TEST SET METRICS =====")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

pd.DataFrame([test_metrics]).to_csv("results/final_test_metrics.csv", index=False)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4.8, 4.2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(f"Confusion Matrix — {final_model_name}\n(Held-out Test Set, n={len(y_test)})")
plt.tight_layout()
plt.savefig("figures/fig2_confusion_matrix.png", dpi=140)
plt.close()

# ROC curve
fig, ax = plt.subplots(figsize=(5, 4.5))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=final_model_name)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC=0.50)")
ax.set_title("ROC Curve — Held-out Test Set")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig3_roc_curve.png", dpi=140)
plt.close()

# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================
clf = final_pipeline.named_steps["clf"]
encoder = final_pipeline.named_steps["encoder"]
raw_feature_names = encoder.get_feature_names_out()
clean_feature_names = [f.replace("cat__", "").replace("num__", "") for f in raw_feature_names]

importances = pd.Series(clf.feature_importances_, index=clean_feature_names).sort_values(ascending=False)
print("\nFeature importances:\n", importances.round(4))
importances.to_csv("results/feature_importances.csv")

fig, ax = plt.subplots(figsize=(7, 5))
importances.sort_values().plot(kind="barh", ax=ax, color="#55A868")
ax.set_xlabel("Feature Importance")
ax.set_title(f"Feature Importance — {final_model_name}")
plt.tight_layout()
plt.savefig("figures/fig4_feature_importance.png", dpi=140)
plt.close()

# ============================================================
# 6. LEARNING CURVE (diagnose over/underfitting)
# ============================================================
train_sizes, train_scores, val_scores = learning_curve(
    final_pipeline, X_train, y_train, cv=cv, scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 8), random_state=42, n_jobs=-1
)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#4C72B0", label="Training F1")
ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color="#4C72B0")
ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color="#C44E52", label="Validation F1")
ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color="#C44E52")
ax.set_xlabel("Training Set Size")
ax.set_ylabel("F1 Score")
ax.set_title(f"Learning Curve — {final_model_name}")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/fig5_learning_curve.png", dpi=140)
plt.close()

print("\nAll figures and result tables saved.")
print("Final model:", final_model_name)
print("Chosen hyperparameters:", grid_rf.best_params_ if final_pipeline is grid_rf.best_estimator_ else grid_gb.best_params_)
