"""Leakage-safe Titanic data preprocessing and feature engineering pipelines.

All stateful transformations (imputation medians, IQR capping bounds,
one-hot categories, class means, and scalers) are fit strictly on training
folds during cross-validation or on the training split, preventing data leakage.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


class TitanicFeaturePipeline(BaseEstimator, TransformerMixin):
    """Stateless and stateful feature engineering for the Titanic dataset.

    Parameters
    ----------
    include_interactions : bool, default=True
        Whether to generate interaction features (FareRelativeToClass,
        AgeClassInteraction) introduced in Week 4. Set to False for the
        baseline 15-feature set from Week 1.
    """

    def __init__(self, include_interactions: bool = True):
        self.include_interactions = include_interactions

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        # 1. Embarked mode
        emb = X["Embarked"].dropna().astype(str).str.strip().str.upper()
        self.embarked_mode_ = emb.mode()[0] if len(emb) > 0 else "S"

        # 2. Title extraction & common titles (freq >= 10 on train)
        titles = (
            X["Name"]
            .astype(str)
            .str.extract(r",\s*([^.]*)\.", expand=False)
            .replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        )
        title_counts = titles.value_counts()
        self.common_titles_ = title_counts[title_counts >= 10].index.tolist()

        # 3. Fare: 0 treated as NaN; median per Pclass; global median fallback
        fare = X["Fare"].astype(float).copy()
        fare[fare == 0] = np.nan
        self.fare_pclass_medians_ = fare.groupby(X["Pclass"]).median().to_dict()
        self.global_fare_median_ = float(fare.median()) if not np.isnan(fare.median()) else 14.45

        # Capping bounds for Fare (k=3.0)
        fare_imp = fare.fillna(X["Pclass"].map(self.fare_pclass_medians_)).fillna(self.global_fare_median_)
        q1_f, q3_f = fare_imp.quantile([0.25, 0.75])
        iqr_f = q3_f - q1_f
        self.fare_low_ = max(0.0, float(q1_f - 3.0 * iqr_f))
        self.fare_high_ = float(q3_f + 3.0 * iqr_f)

        # Class mean fare for FareRelativeToClass (calculated on capped fare)
        fare_capped = fare_imp.clip(self.fare_low_, self.fare_high_)
        self.class_mean_fare_ = fare_capped.groupby(X["Pclass"]).mean().to_dict()

        # 4. Age: grouped median by (Pclass, Sex); global median fallback
        sex_norm = X["Sex"].astype(str).str.strip().str.lower()
        self.age_group_medians_ = X.groupby([X["Pclass"], sex_norm])["Age"].median().to_dict()
        self.global_age_median_ = float(X["Age"].median()) if not np.isnan(X["Age"].median()) else 28.0

        # Capping bounds for Age (k=1.5)
        keys = list(zip(X["Pclass"], sex_norm))
        mapped_age = pd.Series(
            [self.age_group_medians_.get(k, self.global_age_median_) for k in keys],
            index=X.index,
        )
        age_imp = X["Age"].fillna(mapped_age).fillna(self.global_age_median_)
        q1_a, q3_a = age_imp.quantile([0.25, 0.75])
        iqr_a = q3_a - q1_a
        self.age_low_ = max(0.0, float(q1_a - 1.5 * iqr_a))
        self.age_upper_ = float(q3_a + 1.5 * iqr_a)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        out = pd.DataFrame(index=X.index)

        out["Pclass"] = X["Pclass"].astype(int)

        # Sex: male -> 0, female -> 1
        sex_str = X["Sex"].astype(str).str.strip().str.lower()
        out["Sex"] = sex_str.map({"male": 0, "female": 1}).fillna(0).astype(int)

        # HasCabin
        out["HasCabin"] = X["Cabin"].notnull().astype(int)

        # FamilySize & IsAlone
        out["SibSp"] = X["SibSp"].astype(int)
        out["Parch"] = X["Parch"].astype(int)
        out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
        out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

        # Embarked (imputed with mode learned during fit)
        emb = X["Embarked"].astype(str).str.strip().str.upper()
        emb_clean = emb.replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
        out["Embarked"] = emb_clean.fillna(self.embarked_mode_)

        # Title
        titles = (
            X["Name"]
            .astype(str)
            .str.extract(r",\s*([^.]*)\.", expand=False)
            .replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        )
        out["Title"] = titles.where(titles.isin(self.common_titles_), "Rare")

        # Age: impute with learned group medians, then clip to learned bounds
        keys = list(zip(X["Pclass"], sex_str))
        mapped_age = pd.Series(
            [self.age_group_medians_.get(k, self.global_age_median_) for k in keys],
            index=X.index,
        )
        age_imp = X["Age"].astype(float).fillna(mapped_age).fillna(self.global_age_median_)
        out["Age"] = age_imp.clip(self.age_low_, self.age_upper_)

        # Fare: impute 0s/missing with learned class medians, then clip to learned bounds
        fare = X["Fare"].astype(float).copy()
        fare[fare == 0] = np.nan
        fare_imp = fare.fillna(X["Pclass"].map(self.fare_pclass_medians_)).fillna(self.global_fare_median_)
        out["Fare"] = fare_imp.clip(self.fare_low_, self.fare_high_)

        # Interactions
        if self.include_interactions:
            class_avg = X["Pclass"].map(self.class_mean_fare_).fillna(self.global_fare_median_)
            out["FareRelativeToClass"] = out["Fare"] / class_avg
            out["AgeClassInteraction"] = out["Age"] * out["Pclass"]

        return out


def get_feature_column_transformer(include_interactions: bool = True) -> ColumnTransformer:
    """Return a ColumnTransformer for one-hot encoding categoricals and passing numerics."""
    cat_cols = ["Embarked", "Title"]
    num_cols = ["Pclass", "Sex", "HasCabin", "SibSp", "Parch", "FamilySize", "IsAlone", "Age", "Fare"]
    if include_interactions:
        num_cols.extend(["FareRelativeToClass", "AgeClassInteraction"])

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ]
    )


def build_titanic_pipeline(
    classifier,
    needs_scaling: bool = False,
    include_interactions: bool = True,
) -> Pipeline:
    """Build a complete, leakage-safe scikit-learn Pipeline."""
    steps = [
        ("prep", TitanicFeaturePipeline(include_interactions=include_interactions)),
        ("encoder", get_feature_column_transformer(include_interactions=include_interactions)),
    ]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", classifier))
    return Pipeline(steps)
