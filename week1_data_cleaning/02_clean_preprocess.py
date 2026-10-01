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

sns.set_style("whitegrid")
df = pd.read_csv("../data/raw/titanic.csv")
raw_shape = df.shape

# ---------- 1. Missing value visualization (before) ----------
plt.figure(figsize=(7,4))
missing = df.isnull().mean().sort_values(ascending=False) * 100
missing = missing[missing > 0]
sns.barplot(x=missing.values, y=missing.index, color="#4C72B0")
plt.xlabel("% missing")
plt.title("Missing values by column (before cleaning)")
plt.tight_layout()
plt.savefig("figures/fig1_missing_before.png", dpi=140)
plt.close()

# ---------- 2. Outlier visualization (before) - Age & Fare ----------
fig, axes = plt.subplots(1, 2, figsize=(9,4))
sns.boxplot(y=df["Age"], ax=axes[0], color="#55A868")
axes[0].set_title("Age - before")
sns.boxplot(y=df["Fare"], ax=axes[1], color="#C44E52")
axes[1].set_title("Fare - before")
plt.tight_layout()
plt.savefig("figures/fig2_outliers_before.png", dpi=140)
plt.close()

# ---------- 3. Handle missing values ----------
# Cabin: 77% missing -> engineer a binary "HasCabin" flag, drop raw column
df["HasCabin"] = df["Cabin"].notnull().astype(int)
df.drop(columns=["Cabin"], inplace=True)

# Embarked: only 2 missing -> impute with mode
embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)

# Age: 19.9% missing -> impute with median grouped by Pclass & Sex
# (more informative than a single global median)
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda s: s.fillna(s.median())
)
# safety net in case any group had no data at all
df["Age"] = df["Age"].fillna(df["Age"].median())

# ---------- 4. Fix erroneous / inconsistent entries ----------
# Fare == 0 for paying passengers is very likely a data-entry issue
# (crew/employee tickets aside, 0-fare 1st/2nd class passengers are implausible)
zero_fare_mask = df["Fare"] == 0
df.loc[zero_fare_mask, "Fare"] = np.nan
df["Fare"] = df.groupby("Pclass")["Fare"].transform(lambda s: s.fillna(s.median()))

# Standardise categorical text (case/whitespace) even though this dataset is already clean-ish
df["Sex"] = df["Sex"].str.strip().str.lower()
df["Embarked"] = df["Embarked"].str.strip().str.upper()

# ---------- 5. Outlier treatment (IQR capping / winsorizing) on Age & Fare ----------
def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

age_low, age_high = iqr_bounds(df["Age"])
fare_low, fare_high = iqr_bounds(df["Fare"], k=3)  # fare is heavily right-skewed, use wider k

n_age_outliers = ((df["Age"] < age_low) | (df["Age"] > age_high)).sum()
n_fare_outliers = ((df["Fare"] < fare_low) | (df["Fare"] > fare_high)).sum()

df["Age"] = df["Age"].clip(lower=max(age_low, 0), upper=age_high)
df["Fare"] = df["Fare"].clip(lower=max(fare_low, 0), upper=fare_high)

# ---------- 6. Feature engineering ----------
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")
rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
df["Title"] = df["Title"].replace(rare_titles, "Rare")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

df.drop(columns=["Name", "Ticket", "PassengerId"], inplace=True)

# ---------- 7. Encode categoricals ----------
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True, dtype=int)

# ---------- 8. Post-cleaning visuals ----------
fig, axes = plt.subplots(1, 2, figsize=(9,4))
sns.boxplot(y=df["Age"], ax=axes[0], color="#55A868")
axes[0].set_title("Age - after")
sns.boxplot(y=df["Fare"], ax=axes[1], color="#C44E52")
axes[1].set_title("Fare - after")
plt.tight_layout()
plt.savefig("figures/fig3_outliers_after.png", dpi=140)
plt.close()

plt.figure(figsize=(4,3.5))
missing_after = df.isnull().sum().sum()
plt.bar(["Missing cells"], [missing_after], color="#4C72B0")
plt.title(f"Total missing cells after cleaning = {missing_after}")
plt.tight_layout()
plt.savefig("figures/fig4_missing_after.png", dpi=140)
plt.close()

corr_cols = ["Survived","Pclass","Sex","Age","Fare","FamilySize","IsAlone","HasCabin"]
plt.figure(figsize=(6,5))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap (cleaned features)")
plt.tight_layout()
plt.savefig("figures/fig5_correlation.png", dpi=140)
plt.close()

# ---------- 9. Save cleaned dataset ----------
df.to_csv("../data/processed/titanic_cleaned.csv", index=False)

print("RAW SHAPE:", raw_shape)
print("CLEAN SHAPE:", df.shape)
print("Embarked mode used for imputation:", embarked_mode)
print("Zero-fare rows fixed:", zero_fare_mask.sum())
print("Age IQR bounds:", age_low, age_high, "| outliers found:", n_age_outliers)
print("Fare IQR bounds (k=3):", fare_low, fare_high, "| outliers found:", n_fare_outliers)
print("Missing values remaining:\n", df.isnull().sum().sum())
print("\nFinal columns:", list(df.columns))
print("\nSample cleaned rows:\n", df.head())
