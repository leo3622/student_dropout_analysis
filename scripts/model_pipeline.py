from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


TARGET_COL = "Output"
REPORT_PATH = Path("reports/model_report.json")


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features to reduce the influence of extreme values."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.quantile(X_array, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(X_array, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=float)
        return np.clip(X_array, self.lower_bounds_, self.upper_bounds_)


def load_raw_data(path: str = "data/student_data.csv") -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8")


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"Nacionality": "Nationality"})


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "Curricular units 1st sem (grade)" in enriched.columns and "Curricular units 2nd sem (grade)" in enriched.columns:
        enriched["performance_score"] = (
            enriched["Curricular units 1st sem (grade)"] + enriched["Curricular units 2nd sem (grade)"]
        ) / 2
    return enriched


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["Gender"] = cleaned["Gender"].map({0: "Female", 1: "Male"}).fillna(cleaned["Gender"])
    cleaned["evening attendance"] = cleaned["evening attendance"].map({1: "Day time", 0: "Evening time"}).fillna(
        cleaned["evening attendance"]
    )
    return cleaned


def infer_feature_lists(df: pd.DataFrame, target: str = TARGET_COL, max_unique_for_cat: int = 15) -> Tuple[List[str], List[str]]:
    feature_cols = [col for col in df.columns if col != target]
    categorical_cols = [
        col for col in feature_cols if df[col].dtype == object or df[col].nunique(dropna=False) <= max_unique_for_cat
    ]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    return categorical_cols, numeric_cols


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", OutlierClipper()),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_cols),
            ("numeric", numeric_pipeline, numeric_cols),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, classes: Iterable[str]) -> Dict:
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    y_true_binarized = label_binarize(y_test, classes=classes)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(y_true_binarized, probabilities, average="macro", multi_class="ovr"),
    }


def fairness_by_group(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, sensitive_col: str) -> Dict[str, Dict]:
    """Compute precision/recall for the 'Dropout' class across sensitive groups."""
    y_true_binary = (y_test == "Dropout").astype(int)
    preds = model.predict(X_test)
    y_pred_binary = (pd.Series(preds, index=y_test.index) == "Dropout").astype(int)

    metrics: Dict[str, Dict] = {}
    for group_value in sorted(X_test[sensitive_col].unique()):
        mask = X_test[sensitive_col] == group_value
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary[mask],
            y_pred_binary[mask],
            average="binary",
            zero_division=0,
        )
        metrics[str(group_value)] = {
            "support": int(mask.sum()),
            "precision_dropout": float(precision),
            "recall_dropout": float(recall),
            "f1_dropout": float(f1),
        }
    return metrics


def get_feature_names(model: Pipeline, categorical_cols: List[str], numeric_cols: List[str]) -> List[str]:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    ohe: OneHotEncoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    categorical_features = ohe.get_feature_names_out(categorical_cols).tolist()
    return categorical_features + numeric_cols


def feature_importances(model: Pipeline, categorical_cols: List[str], numeric_cols: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    feature_names = get_feature_names(model, categorical_cols, numeric_cols)
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).mean(axis=0)
    else:
        return []

    top_indices = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], float(importances[i])) for i in top_indices]


@dataclass
class ModelResult:
    name: str
    metrics: Dict
    params: Dict
    fairness: Dict[str, Dict] | None
    feature_importance: List[Tuple[str, float]]


def train_models(df: pd.DataFrame) -> Dict[str, ModelResult]:
    df = rename_columns(df)
    df = normalize_categories(df)
    df = add_derived_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    categorical_cols, numeric_cols = infer_feature_lists(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    models = {
        "baseline_dummy": DummyClassifier(strategy="stratified", random_state=42),
        "log_reg": LogisticRegression(max_iter=500, multi_class="auto", class_weight="balanced"),
        "tuned_random_forest": RandomForestClassifier(random_state=42),
    }

    results: Dict[str, ModelResult] = {}

    # Baseline dummy
    baseline_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", models["baseline_dummy"])])
    baseline_pipeline.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_pipeline, X_test, y_test, classes=np.unique(y_train))
    results["baseline_dummy"] = ModelResult(
        name="Baseline stratified dummy",
        metrics=baseline_metrics,
        params={},
        fairness=None,
        feature_importance=[],
    )

    # Logistic regression
    log_reg_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", models["log_reg"])])
    log_reg_pipeline.fit(X_train, y_train)
    log_reg_metrics = evaluate_model(log_reg_pipeline, X_test, y_test, classes=np.unique(y_train))
    log_reg_fairness = {
        "Gender": fairness_by_group(log_reg_pipeline, X_test, y_test, sensitive_col="Gender"),
        "evening attendance": fairness_by_group(log_reg_pipeline, X_test, y_test, sensitive_col="evening attendance"),
    }
    results["log_reg"] = ModelResult(
        name="Regularized logistic regression",
        metrics=log_reg_metrics,
        params=log_reg_pipeline.named_steps["model"].get_params(),
        fairness=log_reg_fairness,
        feature_importance=feature_importances(log_reg_pipeline, categorical_cols, numeric_cols),
    )

    # Tuned Random Forest
    rf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", models["tuned_random_forest"])])
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 12, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__class_weight": [None, "balanced_subsample"],
    }
    grid = GridSearchCV(
        rf_pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model: Pipeline = grid.best_estimator_
    rf_metrics = evaluate_model(best_model, X_test, y_test, classes=np.unique(y_train))
    rf_fairness = {
        "Gender": fairness_by_group(best_model, X_test, y_test, sensitive_col="Gender"),
        "evening attendance": fairness_by_group(best_model, X_test, y_test, sensitive_col="evening attendance"),
    }

    results["tuned_random_forest"] = ModelResult(
        name="Tuned random forest",
        metrics=rf_metrics,
        params=grid.best_params_,
        fairness=rf_fairness,
        feature_importance=feature_importances(best_model, categorical_cols, numeric_cols),
    )

    return results


def serialize_report(results: Dict[str, ModelResult]) -> Dict:
    best_model_name = max(results, key=lambda name: results[name].metrics["f1_macro"])

    return {
        "best_model": best_model_name,
        "models": {
            name: {
                "display_name": result.name,
                "metrics": result.metrics,
                "params": result.params,
                "fairness": result.fairness,
                "feature_importance": result.feature_importance,
            }
            for name, result in results.items()
        },
    }


def run_training() -> Dict:
    df = load_raw_data()
    results = train_models(df)
    report = serialize_report(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    output = run_training()
    print(json.dumps(output, indent=2))
