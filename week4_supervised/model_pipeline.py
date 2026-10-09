import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # run from anywhere; paths below are relative to this folder
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay
)

sns.set_style("whitegrid")
np.random.seed(42)

# ============================================================
# 1. LOAD DATA (already cleaned & feature-engineered in Week 1)
# ============================================================
df = pd.read_csv("../data/processed/titanic_cleaned.csv")
print("Dataset shape:", df.shape)
print("Class balance:\n", df["Survived"].value_counts(normalize=True).round(3))

X = df.drop(columns=["Survived"])
y = df["Survived"]

# ============================================================
# 2. ADDITIONAL FEATURE ENGINEERING FOR MODELLING
# ============================================================
# Interaction feature: class-adjusted fare (fare relative to the mean fare of its own class)
# captures whether a passenger paid a premium/discount relative to peers in the same class,
# which plain Fare cannot express.
class_avg_fare = df.groupby("Pclass")["Fare"].transform("mean")
X["FareRelativeToClass"] = df["Fare"] / class_avg_fare

# Age x Pclass interaction: a child's survival advantage differs by class (seen in Week 2 EDA)
X["AgeClassInteraction"] = df["Age"] * df["Pclass"]

print("\nFinal feature set:", list(X.columns))

# ============================================================
# 3. TRAIN / TEST SPLIT (stratified to preserve class balance)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print("Train class balance:\n", y_train.value_counts(normalize=True).round(3))
print("Test class balance:\n", y_test.value_counts(normalize=True).round(3))

# Feature scaling (needed for Logistic Regression, SVM, KNN; harmless for tree models)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# ============================================================
# 4. MODEL COMPARISON VIA 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM (RBF kernel)": SVC(probability=True, random_state=42),
}

scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_results = []
for name, model in models.items():
    # scaled features for distance/gradient-based models, raw for tree-based models
    Xtr = X_train_scaled if name in ["Logistic Regression", "K-Nearest Neighbors", "SVM (RBF kernel)"] else X_train
    scores = cross_validate(model, Xtr, y_train, cv=cv, scoring=scoring)
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
print("\n===== 5-FOLD CV MODEL COMPARISON (on training set) =====")
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
ax.set_title("Model Comparison — 5-Fold Cross-Validation on Training Set")
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("figures/fig1_model_comparison.png", dpi=140)
plt.close()

# ============================================================
# 5. HYPERPARAMETER TUNING ON THE BEST MODEL (Random Forest / Gradient Boosting)
# ============================================================
best_model_name = cv_df.iloc[0]["Model"]
print(f"\nBest model by CV F1: {best_model_name}")

param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid, cv=cv, scoring="f1", n_jobs=-1
)
grid.fit(X_train, y_train)
print("\nBest hyperparameters (Random Forest):", grid.best_params_)
print("Best CV F1 score:", round(grid.best_score_, 4))

best_rf = grid.best_estimator_

# Also tune Gradient Boosting for comparison since it's often competitive
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {"n_estimators": [100, 200], "max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]},
    cv=cv, scoring="f1", n_jobs=-1
)
gb_grid.fit(X_train, y_train)
print("\nBest hyperparameters (Gradient Boosting):", gb_grid.best_params_)
print("Best CV F1 score:", round(gb_grid.best_score_, 4))

final_model = best_rf if grid.best_score_ >= gb_grid.best_score_ else gb_grid.best_estimator_
final_model_name = "Random Forest (tuned)" if final_model is best_rf else "Gradient Boosting (tuned)"
print(f"\nFinal chosen model: {final_model_name}")

# ============================================================
# 6. FINAL EVALUATION ON HELD-OUT TEST SET
# ============================================================
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

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
ax.set_title(f"Confusion Matrix — {final_model_name}\n(Test Set, n={len(y_test)})")
plt.tight_layout()
plt.savefig("figures/fig2_confusion_matrix.png", dpi=140)
plt.close()

# ROC curve
fig, ax = plt.subplots(figsize=(5, 4.5))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=final_model_name, color="#4C72B0")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC=0.50)")
ax.set_title("ROC Curve — Test Set")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig3_roc_curve.png", dpi=140)
plt.close()

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
importances = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
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
# 8. LEARNING CURVE (diagnose over/underfitting)
# ============================================================
train_sizes, train_scores, val_scores = learning_curve(
    final_model, X_train, y_train, cv=cv, scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 8), random_state=42
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
print("Chosen hyperparameters:", grid.best_params_ if final_model is best_rf else gb_grid.best_params_)
