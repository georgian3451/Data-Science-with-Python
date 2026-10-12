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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, silhouette_score, RocCurveDisplay
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

np.random.seed(42)
sns.set_style("whitegrid")

# ============================================================
# PHASE 1 — DATA ACQUISITION
# ============================================================
raw = pd.read_csv("../data/raw/titanic.csv")
print("Raw shape:", raw.shape)
print("Missing values:\n", raw.isnull().sum())

# ============================================================
# PHASE 2 — DATA CLEANING (condensed from Week 1 methodology)
# ============================================================
df = raw.copy()

df["HasCabin"] = df["Cabin"].notnull().astype(int)
df.drop(columns=["Cabin"], inplace=True)

embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)

df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda s: s.fillna(s.median()))
df["Age"] = df["Age"].fillna(df["Age"].median())

zero_fare = df["Fare"] == 0
df.loc[zero_fare, "Fare"] = np.nan
df["Fare"] = df.groupby("Pclass")["Fare"].transform(lambda s: s.fillna(s.median()))

def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

age_low, age_high = iqr_bounds(df["Age"])
fare_low, fare_high = iqr_bounds(df["Fare"], k=3)
df["Age"] = df["Age"].clip(lower=max(age_low, 0), upper=age_high)
df["Fare"] = df["Fare"].clip(lower=max(fare_low, 0), upper=fare_high)

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")
rare = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
df["Title"] = df["Title"].replace(rare, "Rare")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

df.drop(columns=["Name", "Ticket", "PassengerId"], inplace=True)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df_encoded = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True, dtype=int)

print("\nCleaned shape:", df_encoded.shape)
print("Missing after cleaning:", df_encoded.isnull().sum().sum())
df_encoded.to_csv("../data/processed/capstone_cleaned.csv", index=False)

# ============================================================
# PHASE 3 — EXPLORATORY DATA ANALYSIS
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df.assign(Sex=df["Sex"].map({0:"Male",1:"Female"})),
            ax=axes[0], palette={"Male":"#4C72B0","Female":"#C44E52"})
axes[0].set_title("Survival Rate by Class & Sex")
axes[0].set_xlabel("Passenger Class"); axes[0].set_ylabel("Survival Rate")

sns.histplot(df["Age"], bins=25, kde=True, ax=axes[1], color="#55A868")
axes[1].set_title("Age Distribution (cleaned)")
axes[1].set_xlabel("Age")

corr_cols = ["Survived","Pclass","Sex","Age","Fare","FamilySize","IsAlone","HasCabin"]
sns.heatmap(df_encoded[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[2],
            cbar_kws={"label":"corr"})
axes[2].set_title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("figures/fig1_eda_summary.png", dpi=140)
plt.close()

print("\nOverall survival rate: {:.1f}%".format(df["Survived"].mean()*100))
print("Correlation with Survived:\n", df_encoded[corr_cols].corr()["Survived"].sort_values(ascending=False))

# ============================================================
# PHASE 4 — SUPERVISED MODELING (Predict Survival)
# ============================================================
X = df_encoded.drop(columns=["Survived"])
y = df_encoded["Survived"]

class_avg_fare = df_encoded.groupby("Pclass")["Fare"].transform("mean")
X["FareRelativeToClass"] = df_encoded["Fare"] / class_avg_fare
X["AgeClassInteraction"] = df_encoded["Age"] * df_encoded["Pclass"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {"n_estimators": [100, 200], "max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]}
grid = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("\nBest hyperparameters:", grid.best_params_)
print("Best CV F1:", round(grid.best_score_, 4))

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

sup_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "ROC_AUC": roc_auc_score(y_test, y_proba),
}
print("\nSupervised test metrics:", {k: round(v,4) for k,v in sup_metrics.items()})
print("\n", classification_report(y_test, y_pred, target_names=["Died","Survived"]))

cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Died","Survived"], yticklabels=["Died","Survived"])
axes[0].set_title(f"Confusion Matrix (Test, n={len(y_test)})")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1], name="Gradient Boosting", color="#4C72B0")
axes[1].plot([0,1],[0,1], linestyle="--", color="gray")
axes[1].set_title("ROC Curve")
plt.tight_layout()
plt.savefig("figures/fig2_supervised_evaluation.png", dpi=140)
plt.close()

importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop features:\n", importances.head(8))

fig, ax = plt.subplots(figsize=(7,4.5))
importances.head(10).sort_values().plot(kind="barh", ax=ax, color="#55A868")
ax.set_title("Top 10 Feature Importances — Gradient Boosting")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("figures/fig3_feature_importance.png", dpi=140)
plt.close()

pd.DataFrame([sup_metrics]).to_csv("results/supervised_metrics.csv", index=False)

# ============================================================
# PHASE 5 — UNSUPERVISED MODELING (Passenger Segmentation)
# ============================================================
# Use descriptive, non-target features only (exclude Survived - the point is to find
# structure in the data independent of the label, then check afterward if it aligns with survival)
cluster_features = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "HasCabin"]
X_cluster = df_encoded[cluster_features].copy()

scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Elbow method + silhouette score to choose k
inertias = []
silhouettes = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_cluster_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
axes[0].plot(list(k_range), inertias, "o-", color="#4C72B0")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[1].plot(list(k_range), silhouettes, "o-", color="#C44E52")
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("figures/fig4_cluster_selection.png", dpi=140)
plt.close()

print("\nInertias:", [round(i,1) for i in inertias])
print("Silhouette scores:", [round(s,4) for s in silhouettes])

# Choose k with best silhouette score (report will explain rationale)
best_k = list(k_range)[int(np.argmax(silhouettes))]
print(f"\nChosen k = {best_k} (highest silhouette score = {max(silhouettes):.4f})")

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_cluster_scaled)
df_encoded["Cluster"] = cluster_labels

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster_scaled)
print("\nPCA explained variance ratio:", pca.explained_variance_ratio_, "total:", pca.explained_variance_ratio_.sum())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
scatter1 = axes[0].scatter(X_pca[:,0], X_pca[:,1], c=cluster_labels, cmap="Set1", alpha=0.6, s=18)
axes[0].set_title(f"Passenger Clusters (k={best_k}) in PCA Space")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
legend1 = axes[0].legend(*scatter1.legend_elements(), title="Cluster", loc="best")
axes[0].add_artist(legend1)

scatter2 = axes[1].scatter(X_pca[:,0], X_pca[:,1], c=df_encoded["Survived"], cmap="coolwarm", alpha=0.6, s=18)
axes[1].set_title("Same PCA Space, Colored by Actual Survival")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
legend2 = axes[1].legend(*scatter2.legend_elements(), title="Survived", loc="best")
axes[1].add_artist(legend2)
plt.tight_layout()
plt.savefig("figures/fig5_pca_clusters.png", dpi=140)
plt.close()

# Cluster profiling: mean feature values + survival rate per cluster
cluster_profile = df_encoded.groupby("Cluster").agg(
    Size=("Survived", "size"),
    SurvivalRate=("Survived", "mean"),
    AvgPclass=("Pclass", "mean"),
    PctFemale=("Sex", "mean"),
    AvgAge=("Age", "mean"),
    AvgFare=("Fare", "mean"),
    AvgFamilySize=("FamilySize", "mean"),
    PctHasCabin=("HasCabin", "mean"),
).round(2)
print("\n===== CLUSTER PROFILES =====\n", cluster_profile)
cluster_profile.to_csv("results/cluster_profiles.csv")

fig, ax = plt.subplots(figsize=(7, 4.2))
cluster_profile["SurvivalRate"].plot(kind="bar", ax=ax, color=sns.color_palette("Set1", best_k))
ax.axhline(df["Survived"].mean(), color="black", linestyle="--", label=f"Overall rate ({df['Survived'].mean():.2f})")
ax.set_title("Survival Rate by Discovered Passenger Cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel("Survival Rate")
ax.tick_params(axis="x", rotation=0)
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig6_cluster_survival.png", dpi=140)
plt.close()

print("\nAll figures and tables saved. Capstone pipeline complete.")
print(f"\nFINAL SUPERVISED TEST ACCURACY: {sup_metrics['Accuracy']:.4f}, F1: {sup_metrics['F1']:.4f}, ROC-AUC: {sup_metrics['ROC_AUC']:.4f}")
print(f"FINAL CLUSTERING: k={best_k}, silhouette={max(silhouettes):.4f}")

# ============================================================
# PHASE 5b — SUPPLEMENTARY: richer k=4 segmentation for practical insight
# (silhouette-optimal k=2 is statistically cleanest but coarse; k=4 trades a
# small amount of separation for more actionable, descriptive personas)
# ============================================================
kmeans_k4 = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_k4 = kmeans_k4.fit_predict(X_cluster_scaled)
df_encoded["Cluster_k4"] = labels_k4

profile_k4 = df_encoded.groupby("Cluster_k4").agg(
    Size=("Survived", "size"),
    SurvivalRate=("Survived", "mean"),
    AvgPclass=("Pclass", "mean"),
    PctFemale=("Sex", "mean"),
    AvgAge=("Age", "mean"),
    AvgFare=("Fare", "mean"),
    AvgFamilySize=("FamilySize", "mean"),
    PctHasCabin=("HasCabin", "mean"),
).round(2)
print("\n===== k=4 CLUSTER PROFILES (supplementary) =====\n", profile_k4)
profile_k4.to_csv("results/cluster_profiles_k4.csv")

fig, ax = plt.subplots(figsize=(11, 4.5))
X_pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
scatter = ax.scatter(X_pca_df["PC1"], X_pca_df["PC2"], c=labels_k4, cmap="tab10", alpha=0.65, s=20)
ax.set_title("Supplementary View: k=4 Passenger Segments in PCA Space")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
legend = ax.legend(*scatter.legend_elements(), title="Segment", loc="best")
ax.add_artist(legend)
plt.tight_layout()
plt.savefig("figures/fig7_pca_k4_segments.png", dpi=140)
plt.close()
