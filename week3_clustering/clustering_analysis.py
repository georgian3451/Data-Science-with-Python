"""Week 3 - Unsupervised Learning and Clustering Analysis (UCI Wine dataset).

K-Means (k chosen via elbow + silhouette) validated against Ward hierarchical clustering,
visualised with PCA, profiled per cluster and externally validated against the true cultivars.

Run:  python week3_clustering/clustering_analysis.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # run from anywhere
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_samples, silhouette_score)
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")

# ------------------------------------------------------------------ 1. Data
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y_true = data.target                                   # used ONLY for external validation
os.makedirs("../data/raw", exist_ok=True)
X.assign(cultivar=y_true).to_csv("../data/raw/wine.csv", index=False)
print("Shape:", X.shape, "| missing:", int(X.isnull().sum().sum()))

# ------------------------------------------------------------------ 2. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------ 3. Choose k
ks = list(range(2, 10))
inertias, sil_scores = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
pd.DataFrame({"k": ks, "inertia": inertias, "silhouette": sil_scores}).to_csv("results/k_selection.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertias, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(ks, sil_scores, "o-", color="#DD8452"); ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette")
plt.tight_layout(); plt.savefig("figures/fig1_elbow_silhouette.png", dpi=140); plt.close()
best_k = ks[int(np.argmax(sil_scores))]
print("Silhouette by k:", dict(zip(ks, np.round(sil_scores, 3))), "-> best k =", best_k)

# ------------------------------------------------------------------ 4. K-Means (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
sil_avg = silhouette_score(X_scaled, cluster_labels)
print(f"K-Means k=3 silhouette = {sil_avg:.3f}")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("PCA explained variance:", np.round(pca.explained_variance_ratio_, 3),
      "total =", round(pca.explained_variance_ratio_.sum(), 3))
centroids_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(7, 5.5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette="Set1", s=45)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="black", marker="X", s=200, label="Centroids")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("K-Means clusters (k=3) in PCA space"); plt.legend()
plt.tight_layout(); plt.savefig("figures/fig2_kmeans_pca.png", dpi=140); plt.close()

# per-sample silhouette plot
sil_vals = silhouette_samples(X_scaled, cluster_labels)
plt.figure(figsize=(7, 5.5))
y_lower = 10
for c in range(3):
    v = np.sort(sil_vals[cluster_labels == c])
    plt.fill_betweenx(np.arange(y_lower, y_lower + len(v)), 0, v, alpha=0.8, label=f"Cluster {c}")
    y_lower += len(v) + 10
plt.axvline(sil_avg, color="red", linestyle="--", label=f"Mean = {sil_avg:.3f}")
plt.xlabel("Silhouette coefficient"); plt.ylabel("Samples"); plt.title("Silhouette plot (k=3)"); plt.legend()
plt.tight_layout(); plt.savefig("figures/fig3_silhouette_plot.png", dpi=140); plt.close()

# ------------------------------------------------------------------ 5. Hierarchical clustering
Z = linkage(X_scaled, method="ward")
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
agg_labels = agg.fit_predict(X_scaled)
agg_sil = silhouette_score(X_scaled, agg_labels)
print(f"Agglomerative (Ward) silhouette = {agg_sil:.3f}")

plt.figure(figsize=(9, 4.5))
dendrogram(Z, truncate_mode="lastp", p=20, leaf_rotation=90., show_contracted=True)
plt.title("Ward dendrogram (last 20 merges)"); plt.xlabel("Cluster size / sample"); plt.ylabel("Distance")
plt.tight_layout(); plt.savefig("figures/fig4_dendrogram.png", dpi=140); plt.close()

# agreement between the two algorithms (labels are arbitrary -> Hungarian matching)
cm = pd.crosstab(cluster_labels, agg_labels).values
r, c = linear_sum_assignment(-cm)
agreement = cm[r, c].sum()
print(f"K-Means vs Agglomerative agreement: {agreement}/{len(X)} ({agreement/len(X)*100:.0f}%)")

# ------------------------------------------------------------------ 6. Cluster profiling
profile = X.assign(cluster=cluster_labels).groupby("cluster").mean().round(2)
profile.insert(0, "n", pd.Series(cluster_labels).value_counts().sort_index())
profile.T.to_csv("results/cluster_profiles.csv")
print(profile.T.loc[["n", "alcohol", "malic_acid", "total_phenols", "flavanoids",
                     "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"]])

top = ["alcohol", "malic_acid", "flavanoids", "color_intensity", "proline", "od280/od315_of_diluted_wines"]
std_means = pd.DataFrame(X_scaled, columns=X.columns).assign(cluster=cluster_labels).groupby("cluster")[top].mean()
std_means.T.plot.bar(figsize=(9, 4.5), colormap="Set1")
plt.ylabel("Standardised mean"); plt.title("Most discriminating features per cluster"); plt.xticks(rotation=30, ha="right")
plt.tight_layout(); plt.savefig("figures/fig5_cluster_profiles.png", dpi=140); plt.close()

# ------------------------------------------------------------------ 7. External validation
ari = adjusted_rand_score(y_true, cluster_labels)
nmi = normalized_mutual_info_score(y_true, cluster_labels)
print(f"ARI = {ari:.3f} | NMI = {nmi:.3f}")
pd.DataFrame({"metric": ["Silhouette (k=3)", "Agglomerative silhouette", "ARI vs cultivar", "NMI vs cultivar",
                         "KMeans-Agglomerative agreement"],
              "value": [round(sil_avg, 3), round(agg_sil, 3), round(ari, 3), round(nmi, 3), f"{agreement}/{len(X)}"]}
             ).to_csv("results/validation_metrics.csv", index=False)
