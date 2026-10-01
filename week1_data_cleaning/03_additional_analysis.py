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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer

sns.set_style("whitegrid")
raw = pd.read_csv("../data/raw/titanic.csv")
clean = pd.read_csv("../data/processed/titanic_cleaned.csv")

# ============================================================
# A. Deeper univariate / bivariate EDA on the RAW data
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
sns.histplot(raw["Age"].dropna(), bins=30, kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("Age distribution (raw, non-null)")
sns.histplot(raw["Fare"], bins=40, kde=True, ax=axes[1], color="#DD8452")
axes[1].set_title("Fare distribution (raw)")
plt.tight_layout()
plt.savefig("figures/fig6_age_fare_dist.png", dpi=140)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
sns.barplot(x="Sex", y="Survived", data=raw, ax=axes[0], color="#55A868")
axes[0].set_title("Survival rate by Sex")
sns.barplot(x="Pclass", y="Survived", data=raw, ax=axes[1], color="#C44E52")
axes[1].set_title("Survival rate by Pclass")
sns.barplot(x="Embarked", y="Survived", data=raw, ax=axes[2], color="#8172B2")
axes[2].set_title("Survival rate by Embarked")
plt.tight_layout()
plt.savefig("figures/fig7_survival_breakdowns.png", dpi=140)
plt.close()

# ============================================================
# B. Missingness pattern check (is Age missing at random w.r.t. Pclass/Survived?)
# ============================================================
raw["AgeMissing"] = raw["Age"].isnull().astype(int)
missing_by_pclass = raw.groupby("Pclass")["AgeMissing"].mean() * 100
missing_by_survived = raw.groupby("Survived")["AgeMissing"].mean() * 100

# Chi-square test: is Age-missingness independent of Pclass?
ct = pd.crosstab(raw["Pclass"], raw["AgeMissing"])
chi2, p_val, dof, _ = stats.chi2_contingency(ct)

plt.figure(figsize=(5, 3.6))
missing_by_pclass.plot(kind="bar", color="#4C72B0")
plt.ylabel("% Age missing")
plt.title(f"Age missingness by Pclass (chi2 p={p_val:.4f})")
plt.tight_layout()
plt.savefig("figures/fig8_missingness_pattern.png", dpi=140)
plt.close()

# ============================================================
# C. Outlier detection method comparison: IQR vs Z-score on Fare
# ============================================================
fare = raw["Fare"].dropna()
z_scores = np.abs(stats.zscore(fare))
z_outliers = (z_scores > 3).sum()

q1, q3 = fare.quantile([0.25, 0.75])
iqr = q3 - q1
iqr15_outliers = ((fare < q1 - 1.5 * iqr) | (fare > q3 + 1.5 * iqr)).sum()
iqr3_outliers = ((fare < q1 - 3 * iqr) | (fare > q3 + 3 * iqr)).sum()

outlier_method_summary = pd.DataFrame({
    "Method": ["Z-score (|z| > 3)", "IQR (k=1.5)", "IQR (k=3, used)"],
    "Fare outliers flagged": [z_outliers, iqr15_outliers, iqr3_outliers],
    "% of rows": [round(z_outliers/len(fare)*100, 2),
                  round(iqr15_outliers/len(fare)*100, 2),
                  round(iqr3_outliers/len(fare)*100, 2)],
})

# ============================================================
# D. Quantifying the impact of cleaning: naive vs cleaned model
# ============================================================
# Naive pipeline: minimal handling (drop Cabin/Name/Ticket, drop rows with any remaining NA,
# just label-encode Sex/Embarked) - representative of "skip proper cleaning"
naive = raw.drop(columns=["Cabin", "Name", "Ticket", "PassengerId", "AgeMissing"]).copy()
naive["Sex"] = naive["Sex"].map({"male": 0, "female": 1})
naive = pd.get_dummies(naive, columns=["Embarked"], drop_first=True)
naive_dropna = naive.dropna()  # naive approach: just drop incomplete rows

X_naive = naive_dropna.drop(columns=["Survived"])
y_naive = naive_dropna["Survived"]

X_clean = clean.drop(columns=["Survived"])
y_clean = clean["Survived"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000)

naive_scores = cross_val_score(model, X_naive, y_naive, cv=cv, scoring="accuracy")
clean_scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring="accuracy")

model_comparison = pd.DataFrame({
    "Pipeline": ["Naive (rows with any NA dropped)", "Cleaned (this report's pipeline)"],
    "Rows used": [len(X_naive), len(X_clean)],
    "Features": [X_naive.shape[1], X_clean.shape[1]],
    "Mean CV accuracy": [round(naive_scores.mean(), 4), round(clean_scores.mean(), 4)],
    "Std CV accuracy": [round(naive_scores.std(), 4), round(clean_scores.std(), 4)],
})

plt.figure(figsize=(4.5, 3.6))
plt.bar(["Naive pipeline", "Cleaned pipeline"],
        [naive_scores.mean(), clean_scores.mean()],
        yerr=[naive_scores.std(), clean_scores.std()],
        color=["#C44E52", "#55A868"], capsize=6)
plt.ylim(0.5, 0.9)
plt.ylabel("5-fold CV accuracy")
plt.title("Impact of cleaning on model accuracy\n(Logistic Regression)")
plt.tight_layout()
plt.savefig("figures/fig9_model_impact.png", dpi=140)
plt.close()

# ============================================================
# Print everything needed for the report
# ============================================================
print("Missing Age % by Pclass:\n", missing_by_pclass.round(2))
print("\nMissing Age % by Survived:\n", missing_by_survived.round(2))
print(f"\nChi-square test (AgeMissing ~ Pclass): chi2={chi2:.3f}, p={p_val:.5f}, dof={dof}")

print("\nOutlier method comparison:\n", outlier_method_summary)

print("\nNaive pipeline rows retained:", len(X_naive), "of", len(raw))
print("Cleaned pipeline rows retained:", len(X_clean), "of", len(raw))
print("\nModel comparison:\n", model_comparison)
print(f"\nAccuracy delta (cleaned - naive): {clean_scores.mean() - naive_scores.mean():.4f}")

outlier_method_summary.to_csv("results/outlier_method_summary.csv", index=False)
model_comparison.to_csv("results/model_comparison.csv", index=False)
