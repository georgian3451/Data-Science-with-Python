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
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"

# Use the cleaned dataset produced in Week 1, plus the raw file for a couple of
# human-readable categorical labels (Embarked letters, Sex text) that are more
# intuitive to visualize than the one-hot-encoded Week 1 output.
raw = pd.read_csv("../data/raw/titanic.csv")
clean = pd.read_csv("../data/processed/titanic_cleaned.csv")

# Rebuild a "readable" analysis frame: cleaned numeric quality (no missing values,
# outliers capped) but with human-readable category labels for plotting.
df = clean.copy()
df["Sex_label"] = df["Sex"].map({0: "Male", 1: "Female"})
df["Embarked_label"] = np.select(
    [df["Embarked_Q"] == 1, df["Embarked_S"] == 1],
    ["Queenstown", "Southampton"],
    default="Cherbourg",
)
df["Survived_label"] = df["Survived"].map({0: "Died", 1: "Survived"})
df["AgeGroup"] = pd.cut(
    df["Age"], bins=[0, 12, 18, 35, 60, 100],
    labels=["Child (0-12)", "Teen (13-18)", "Adult (19-35)", "Middle Age (36-60)", "Senior (60+)"]
)
df["FareBand"] = pd.qcut(df["Fare"], 4, labels=["Low", "Medium", "High", "Very High"])

print("========== 1. BASIC STATISTICS ==========")
print(df.shape)
print(df[["Age", "Fare", "FamilySize", "Pclass"]].describe().round(2))
print("\nSurvival rate overall: {:.1f}%".format(df["Survived"].mean() * 100))
print("\nValue counts - Pclass:\n", df["Pclass"].value_counts())
print("\nValue counts - Sex:\n", df["Sex_label"].value_counts())
print("\nValue counts - Embarked:\n", df["Embarked_label"].value_counts())
print("\nValue counts - AgeGroup:\n", df["AgeGroup"].value_counts())

# ============================================================
# UNIVARIATE ANALYSIS
# ============================================================

# Fig 1: Survival count + rate
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
sns.countplot(x="Survived_label", data=df, ax=axes[0], palette=["#C44E52", "#55A868"])
axes[0].set_title("Passenger Survival Counts")
axes[0].set_xlabel("Outcome")
axes[0].set_ylabel("Number of Passengers")
for c in axes[0].containers:
    axes[0].bar_label(c)

df["Pclass"].value_counts().sort_index().plot(
    kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452", "#55A868"]
)
axes[1].set_title("Passenger Count by Class")
axes[1].set_xlabel("Passenger Class")
axes[1].set_ylabel("Number of Passengers")
axes[1].set_xticklabels(["1st", "2nd", "3rd"], rotation=0)
plt.tight_layout()
plt.savefig("figures/eda1_survival_class_counts.png", dpi=140)
plt.close()

# Fig 2: Age & Fare distributions
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
sns.histplot(df["Age"], bins=25, kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("Age Distribution")
axes[0].set_xlabel("Age (years)")
axes[0].set_ylabel("Number of Passengers")
sns.histplot(df["Fare"], bins=25, kde=True, ax=axes[1], color="#DD8452")
axes[1].set_title("Fare Distribution")
axes[1].set_xlabel("Fare (capped, GBP)")
axes[1].set_ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("figures/eda2_age_fare_dist.png", dpi=140)
plt.close()

# Fig 3: Categorical breakdown - Sex, Embarked, FamilySize
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
sns.countplot(x="Sex_label", data=df, ax=axes[0], color="#8172B2")
axes[0].set_title("Passengers by Sex")
axes[0].set_xlabel("Sex"); axes[0].set_ylabel("Count")
sns.countplot(x="Embarked_label", data=df, ax=axes[1], color="#937860",
              order=["Southampton", "Cherbourg", "Queenstown"])
axes[1].set_title("Passengers by Port")
axes[1].set_xlabel("Port of Embarkation"); axes[1].set_ylabel("Count")
axes[1].tick_params(axis='x', rotation=15)
sns.countplot(x="FamilySize", data=df, ax=axes[2], color="#64B5CD")
axes[2].set_title("Family Size Aboard")
axes[2].set_xlabel("Family Size"); axes[2].set_ylabel("Count")
plt.tight_layout()
plt.savefig("figures/eda3_categorical_counts.png", dpi=140)
plt.close()

# ============================================================
# BIVARIATE ANALYSIS (relationship with Survived)
# ============================================================

# Fig 4: Survival rate by Sex and Pclass (grouped bar with legend)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
sns.barplot(x="Pclass", y="Survived", hue="Sex_label", data=df, ax=ax,
            palette={"Male": "#4C72B0", "Female": "#C44E52"})
ax.set_title("Survival Rate by Passenger Class and Sex")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Rate")
ax.set_xticklabels(["1st", "2nd", "3rd"])
ax.legend(title="Sex")
plt.tight_layout()
plt.savefig("figures/eda4_survival_by_class_sex.png", dpi=140)
plt.close()

# Fig 5: Age distribution split by survival outcome
fig, ax = plt.subplots(figsize=(6.5, 4.2))
sns.kdeplot(data=df, x="Age", hue="Survived_label", fill=True, common_norm=False,
            palette={"Died": "#C44E52", "Survived": "#55A868"}, alpha=0.45, ax=ax)
ax.set_title("Age Distribution by Survival Outcome")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Density")
plt.tight_layout()
plt.savefig("figures/eda5_age_by_survival.png", dpi=140)
plt.close()

# Fig 6: Survival rate by Age Group
fig, ax = plt.subplots(figsize=(6.5, 4.2))
order = ["Child (0-12)", "Teen (13-18)", "Adult (19-35)", "Middle Age (36-60)", "Senior (60+)"]
sns.barplot(x="AgeGroup", y="Survived", data=df, order=order, ax=ax, color="#55A868")
ax.set_title("Survival Rate by Age Group")
ax.set_xlabel("Age Group")
ax.set_ylabel("Survival Rate")
ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig("figures/eda6_survival_by_agegroup.png", dpi=140)
plt.close()

# Fig 7: Fare vs Survival (boxplot) + Fare band survival rate
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
sns.boxplot(x="Survived_label", y="Fare", data=df, ax=axes[0],
            palette=["#C44E52", "#55A868"])
axes[0].set_title("Fare Distribution by Survival Outcome")
axes[0].set_xlabel("Outcome"); axes[0].set_ylabel("Fare")
sns.barplot(x="FareBand", y="Survived", data=df, ax=axes[1], color="#4C72B0",
            order=["Low", "Medium", "High", "Very High"])
axes[1].set_title("Survival Rate by Fare Band")
axes[1].set_xlabel("Fare Band (quartiles)"); axes[1].set_ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("figures/eda7_fare_survival.png", dpi=140)
plt.close()

# Fig 8: Survival by FamilySize / IsAlone
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
sns.barplot(x="FamilySize", y="Survived", data=df, ax=axes[0], color="#8172B2")
axes[0].set_title("Survival Rate by Family Size")
axes[0].set_xlabel("Family Size (incl. self)"); axes[0].set_ylabel("Survival Rate")
sns.barplot(x="IsAlone", y="Survived", data=df, ax=axes[1], color="#DD8452")
axes[1].set_title("Survival Rate: Alone vs With Family")
axes[1].set_xlabel("Is Alone (0=No, 1=Yes)"); axes[1].set_ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("figures/eda8_family_survival.png", dpi=140)
plt.close()

# Fig 9: Survival by Port of Embarkation and Title
title_cols = [c for c in df.columns if c.startswith("Title_")]
df["Title"] = df[title_cols].idxmax(axis=1).str.replace("Title_", "")
df.loc[df[title_cols].sum(axis=1) == 0, "Title"] = "Master"  # baseline dropped column

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
sns.barplot(x="Embarked_label", y="Survived", data=df, ax=axes[0], color="#64B5CD",
            order=["Southampton", "Cherbourg", "Queenstown"])
axes[0].set_title("Survival Rate by Port of Embarkation")
axes[0].set_xlabel("Port"); axes[0].set_ylabel("Survival Rate")
axes[0].tick_params(axis='x', rotation=15)
sns.barplot(x="Title", y="Survived", data=df, ax=axes[1], color="#937860",
            order=["Mr", "Mrs", "Miss", "Master", "Rare"])
axes[1].set_title("Survival Rate by Title")
axes[1].set_xlabel("Title"); axes[1].set_ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("figures/eda9_port_title_survival.png", dpi=140)
plt.close()

# ============================================================
# MULTIVARIATE ANALYSIS
# ============================================================

# Fig 10: Correlation heatmap (numeric features)
corr_cols = ["Survived", "Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone", "HasCabin"]
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            cbar_kws={"label": "Pearson correlation"})
ax.set_title("Correlation Heatmap of Key Numeric Features")
plt.tight_layout()
plt.savefig("figures/eda10_correlation_heatmap.png", dpi=140)
plt.close()

# Fig 11: Pairplot (Age, Fare, Pclass, FamilySize) colored by Survived
pp = sns.pairplot(
    df[["Age", "Fare", "Pclass", "FamilySize", "Survived_label"]],
    hue="Survived_label", palette={"Died": "#C44E52", "Survived": "#55A868"},
    plot_kws={"alpha": 0.6, "s": 25}, diag_kind="kde", height=1.9
)
pp.fig.suptitle("Pairwise Relationships Colored by Survival Outcome", y=1.02)
pp.savefig("figures/eda11_pairplot.png", dpi=140)
plt.close()

# Fig 12: Heatmap of survival rate across Pclass x AgeGroup (multivariate crosstab)
pivot = df.pivot_table(index="AgeGroup", columns="Pclass", values="Survived", aggfunc="mean", observed=False)
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Survival rate"})
ax.set_title("Survival Rate by Age Group and Passenger Class")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Age Group")
plt.tight_layout()
plt.savefig("figures/eda12_agegroup_class_heatmap.png", dpi=140)
plt.close()

# Fig 13: Violin plot - Age by Class, split by Survival
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.violinplot(x="Pclass", y="Age", hue="Survived_label", data=df, split=True, ax=ax,
               palette={"Died": "#C44E52", "Survived": "#55A868"})
ax.set_title("Age Distribution by Class, Split by Survival Outcome")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Age (years)")
ax.set_xticklabels(["1st", "2nd", "3rd"])
ax.legend(title="Outcome")
plt.tight_layout()
plt.savefig("figures/eda13_violin_age_class_survival.png", dpi=140)
plt.close()

# ============================================================
# AGGREGATIONS / GROUPBY TABLES
# ============================================================
agg1 = df.groupby(["Pclass", "Sex_label"], observed=False).agg(
    Passengers=("Survived", "size"),
    SurvivalRate=("Survived", "mean"),
    AvgAge=("Age", "mean"),
    AvgFare=("Fare", "mean"),
).round(2).reset_index()
print("\n========== GROUPBY: Pclass x Sex ==========\n", agg1)

agg2 = df.groupby("Embarked_label", observed=False).agg(
    Passengers=("Survived", "size"),
    SurvivalRate=("Survived", "mean"),
    AvgFare=("Fare", "mean"),
    Pct1stClass=("Pclass", lambda s: (s == 1).mean() * 100),
).round(2).reset_index()
print("\n========== GROUPBY: Embarked ==========\n", agg2)

agg1.to_csv("results/agg_pclass_sex.csv", index=False)
agg2.to_csv("results/agg_embarked.csv", index=False)

print("\nDone. Figures and aggregation tables saved.")
