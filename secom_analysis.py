# -*- coding: utf-8 -*-
"""End-to-end SECOM coursework analysis in Python.

The script loads the original SECOM files, performs EDA, builds plots,
runs model selection for several machine-learning hypotheses, evaluates the
chosen model on a hold-out split, and writes a Russian Markdown report.

Run example:
    python secom_analysis.py

Optional:
    python secom_analysis.py --out-dir secom_python_outputs --n-jobs -1
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.feature_selection import VarianceThreshold


DEFAULT_DATA_PATH = Path(r"C:\Users\shega\Documents\GitHub\ML_coursework_2026\data\secom.data")
DEFAULT_LABELS_PATH = Path(r"C:\Users\shega\Documents\GitHub\ML_coursework_2026\data\secom_labels.data")
DEFAULT_NAMES_PATH = Path(r"C:\Users\shega\Documents\GitHub\ML_coursework_2026\data\secom.names")
RANDOM_STATE = 20260429


class SensorFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds row-level missingness and robust z-score aggregate indicators.

    Statistics are learned only on the training fold, so the transformer is safe
    inside cross-validation pipelines.
    """

    def fit(self, X: Any, y: Any = None) -> "SensorFeatureEngineer":
        x = np.asarray(X, dtype=float)
        self.n_features_in_ = x.shape[1]
        medians = np.nanmedian(x, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        imputed = np.where(np.isnan(x), medians, x)
        means = imputed.mean(axis=0)
        stds = imputed.std(axis=0)
        stds = np.where(stds > 1e-12, stds, 1.0)
        self.medians_ = medians
        self.means_ = means
        self.stds_ = stds
        return self

    def transform(self, X: Any) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        missing_count = np.isnan(x).sum(axis=1).reshape(-1, 1)
        missing_rate = missing_count / max(1, x.shape[1])
        imputed = np.where(np.isnan(x), self.medians_, x)
        z = (imputed - self.means_) / self.stds_
        mean_abs_z = np.mean(np.abs(z), axis=1).reshape(-1, 1)
        max_abs_z = np.max(np.abs(z), axis=1).reshape(-1, 1)
        return np.hstack([x, missing_count, missing_rate, mean_abs_z, max_abs_z])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            input_features = [f"sensor_{i:03d}" for i in range(self.n_features_in_)]
        return np.asarray(list(input_features) + ["missing_count", "missing_rate", "mean_abs_z", "max_abs_z"])


class MissingRateDropper(BaseEstimator, TransformerMixin):
    """Drops columns with too many missing values in the current training fold."""

    def __init__(self, max_missing_rate: float = 0.50):
        self.max_missing_rate = max_missing_rate

    def fit(self, X: Any, y: Any = None) -> "MissingRateDropper":
        x = np.asarray(X, dtype=float)
        missing_rate = np.isnan(x).mean(axis=0)
        self.keep_mask_ = missing_rate <= self.max_missing_rate
        if not np.any(self.keep_mask_):
            raise ValueError("All features were removed by MissingRateDropper.")
        return self

    def transform(self, X: Any) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        return x[:, self.keep_mask_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            input_features = [f"feature_{i}" for i in range(len(self.keep_mask_))]
        return np.asarray(input_features)[self.keep_mask_]


@dataclass
class SearchResult:
    algorithm: str
    search: GridSearchCV
    best_score: float
    best_params: dict[str, Any]
    best_estimator: Pipeline


def load_secom(data_path: Path, labels_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    x = pd.read_csv(data_path, sep=r"\s+", header=None, na_values=["NaN"], engine="python")
    x.columns = [f"sensor_{i:03d}" for i in range(x.shape[1])]

    records: list[dict[str, Any]] = []
    pattern = re.compile(r'^\s*(-?1)\s+"([^"]+)"\s*$')
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = pattern.match(line)
        if match is None:
            raise ValueError(f"Cannot parse label line: {line}")
        original_label = int(match.group(1))
        records.append(
            {
                "target_original": original_label,
                "target": 1 if original_label == 1 else 0,
                "timestamp": pd.to_datetime(match.group(2), format="%d/%m/%Y %H:%M:%S"),
            }
        )

    labels = pd.DataFrame.from_records(records)
    if len(x) != len(labels):
        raise ValueError(f"Feature rows ({len(x)}) do not match labels ({len(labels)}).")

    return x, labels["target"], labels


def read_dataset_description(names_path: Path) -> str:
    if not names_path.exists():
        return ""
    text = names_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Data Set Information:(.*?)(Using feature selection techniques|Attribute Information:)", text, re.S)
    if match:
        return " ".join(match.group(1).split())
    return " ".join(text.split()[:120])


def cohen_d(pass_values: pd.Series, fail_values: pd.Series) -> float:
    a = pass_values.dropna().to_numpy(dtype=float)
    b = fail_values.dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_var = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2)
    if pooled_var <= 1e-12:
        return np.nan
    return float((np.mean(b) - np.mean(a)) / np.sqrt(pooled_var))


def compute_feature_statistics(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pass_mask = y == 0
    fail_mask = y == 1
    for column in x.columns:
        values = x[column]
        d_signed = cohen_d(values[pass_mask], values[fail_mask])
        rows.append(
            {
                "feature": column,
                "missing_count": int(values.isna().sum()),
                "missing_rate": float(values.isna().mean()),
                "mean": float(values.mean(skipna=True)),
                "std": float(values.std(skipna=True)),
                "min": float(values.min(skipna=True)),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median(skipna=True)),
                "q75": float(values.quantile(0.75)),
                "max": float(values.max(skipna=True)),
                "pass_mean": float(values[pass_mask].mean(skipna=True)),
                "fail_mean": float(values[fail_mask].mean(skipna=True)),
                "cohen_signed": d_signed,
                "cohen_abs": abs(d_signed) if np.isfinite(d_signed) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("cohen_abs", ascending=False, na_position="last")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_target_distribution_plot(y: pd.Series, out_dir: Path) -> Path:
    counts = y.map({0: "Pass (-1)", 1: "Fail (1)"}).value_counts().reindex(["Pass (-1)", "Fail (1)"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(counts.index, counts.values, color=["#2563eb", "#dc2626"])
    ax.set_title("Distribution of target variable")
    ax.set_ylabel("Objects")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value}\n{value / len(y):.1%}", ha="center", va="bottom")
    fig.tight_layout()
    path = out_dir / "target_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_missingness_plot(feature_stats: pd.DataFrame, out_dir: Path) -> Path:
    top = feature_stats.sort_values("missing_rate", ascending=False).head(20).sort_values("missing_rate")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["missing_rate"], color="#7c3aed")
    ax.set_title("Top-20 features by missing value rate")
    ax.set_xlabel("Missing value rate")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "missingness_top20.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_feature_importance_plot(feature_stats: pd.DataFrame, out_dir: Path) -> Path:
    top = feature_stats.dropna(subset=["cohen_abs"]).head(20).sort_values("cohen_abs")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["cohen_abs"], color="#dc2626")
    ax.set_title("Top-20 feature differences between Pass and Fail")
    ax.set_xlabel("|Cohen's d|")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "top_feature_importance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_target_vs_features_plot(x: pd.DataFrame, y: pd.Series, feature_stats: pd.DataFrame, out_dir: Path) -> Path:
    top_features = feature_stats.dropna(subset=["cohen_abs"]).head(5)["feature"].tolist()
    fig, axes = plt.subplots(len(top_features), 1, figsize=(8, 2.6 * len(top_features)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, feature in zip(axes, top_features):
        pass_values = x.loc[y == 0, feature].dropna()
        fail_values = x.loc[y == 1, feature].dropna()
        ax.boxplot([pass_values, fail_values], labels=["Pass (-1)", "Fail (1)"], showfliers=False, patch_artist=True)
        ax.set_title(f"{feature}: distribution by target")
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "target_vs_top_features.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("engineer", SensorFeatureEngineer()),
            ("missing_filter", MissingRateDropper(max_missing_rate=0.50)),
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=40)),
            ("model", model),
        ]
    )


def candidate_models(n_jobs: int) -> list[tuple[str, Pipeline, dict[str, list[Any]]]]:
    return [
        (
            "logistic_regression",
            make_pipeline(LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, random_state=RANDOM_STATE)),
            {
                "select__k": [20, 40, 80, 120],
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l1", "l2"],
            },
        ),
        (
            "gaussian_naive_bayes",
            make_pipeline(GaussianNB()),
            {
                "select__k": [10, 20, 40, 80],
                "model__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3],
            },
        ),
        (
            "knn",
            make_pipeline(KNeighborsClassifier()),
            {
                "select__k": [10, 20, 40],
                "model__n_neighbors": [3, 7, 15, 25],
                "model__weights": ["uniform", "distance"],
            },
        ),
        (
            "decision_tree",
            make_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
            {
                "select__k": [10, 20, 40, 80],
                "model__max_depth": [2, 3, 4, 5],
                "model__min_samples_leaf": [10, 20, 40],
            },
        ),
        (
            "random_forest",
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=250,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=max(1, n_jobs),
                )
            ),
            {
                "select__k": [40, 80],
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [5, 20],
                "model__max_features": ["sqrt"],
            },
        ),
    ]


def run_model_selection(x_train: pd.DataFrame, y_train: pd.Series, n_jobs: int) -> tuple[list[SearchResult], pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "recall_fail": "recall",
        "precision_fail": "precision",
        "f1_fail": "f1",
        "roc_auc": "roc_auc",
    }
    results: list[SearchResult] = []
    result_frames: list[pd.DataFrame] = []

    for algorithm, estimator, param_grid in candidate_models(n_jobs=n_jobs):
        print(f"GridSearchCV: {algorithm}")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            refit="balanced_accuracy",
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        results.append(
            SearchResult(
                algorithm=algorithm,
                search=search,
                best_score=float(search.best_score_),
                best_params=dict(search.best_params_),
                best_estimator=search.best_estimator_,
            )
        )

        frame = pd.DataFrame(search.cv_results_)
        frame["algorithm"] = algorithm
        frame["params_readable"] = frame["params"].map(json.dumps)
        keep_cols = [
            "algorithm",
            "params_readable",
            "mean_test_balanced_accuracy",
            "std_test_balanced_accuracy",
            "mean_test_recall_fail",
            "mean_test_precision_fail",
            "mean_test_f1_fail",
            "mean_test_roc_auc",
            "rank_test_balanced_accuracy",
        ]
        result_frames.append(frame[keep_cols])

    all_results = pd.concat(result_frames, ignore_index=True)
    all_results = all_results.sort_values(
        ["mean_test_balanced_accuracy", "mean_test_f1_fail", "mean_test_recall_fail"],
        ascending=False,
    )
    results.sort(key=lambda item: item.best_score, reverse=True)
    return results, all_results


def choose_threshold_by_cv(best_estimator: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, n_jobs: int) -> tuple[float, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    scores = cross_val_predict(best_estimator, x_train, y_train, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]
    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (scores >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "balanced_accuracy": balanced_accuracy_score(y_train, pred),
                "recall_fail": recall_score(y_train, pred, zero_division=0),
                "precision_fail": precision_score(y_train, pred, zero_division=0),
                "f1_fail": f1_score(y_train, pred, zero_division=0),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["balanced_accuracy", "f1_fail"], ascending=False)
    return float(frame.iloc[0]["threshold"]), frame


def evaluate_at_threshold(model: Pipeline, x: pd.DataFrame, y: pd.Series, threshold: float) -> dict[str, Any]:
    scores = model.predict_proba(x)[:, 1]
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "ber": 1.0 - balanced_accuracy_score(y, pred),
        "recall_fail": recall_score(y, pred, zero_division=0),
        "recall_pass": recall_score(y, pred, pos_label=0, zero_division=0),
        "precision_fail": precision_score(y, pred, zero_division=0),
        "f1_fail": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "classification_report": classification_report(y, pred, target_names=["Pass", "Fail"], zero_division=0),
    }


def selected_feature_names(fitted_pipeline: Pipeline, input_names: list[str]) -> list[str]:
    names: Any = np.asarray(input_names)
    for step_name in ["engineer", "missing_filter", "imputer", "variance", "scaler", "select"]:
        step = fitted_pipeline.named_steps[step_name]
        if hasattr(step, "get_feature_names_out"):
            try:
                names = step.get_feature_names_out(names)
            except TypeError:
                names = step.get_feature_names_out()
    return [str(name) for name in names]


def build_surrogate(
    best_estimator: Pipeline,
    threshold: float,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    out_dir: Path,
) -> tuple[DecisionTreeClassifier, float, str, Path]:
    preprocess = best_estimator[:-1]
    transformed = preprocess.transform(x_train)
    best_scores = best_estimator.predict_proba(x_train)[:, 1]
    best_pred = (best_scores >= threshold).astype(int)
    surrogate = DecisionTreeClassifier(max_depth=3, min_samples_leaf=35, class_weight="balanced", random_state=RANDOM_STATE)
    surrogate.fit(transformed, best_pred)
    fidelity = accuracy_score(best_pred, surrogate.predict(transformed))

    rules = export_text(surrogate, feature_names=feature_names, decimals=3)
    rules_path = out_dir / "surrogate_rules.txt"
    rules_path.write_text(rules, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        surrogate,
        feature_names=feature_names,
        class_names=["Predicted Pass", "Predicted Fail"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    fig.tight_layout()
    plot_path = out_dir / "surrogate_tree.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return surrogate, float(fidelity), rules, plot_path


def save_model_comparison_plot(best_by_algorithm: pd.DataFrame, out_dir: Path) -> Path:
    frame = best_by_algorithm.sort_values("mean_test_balanced_accuracy")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(frame["algorithm"], frame["mean_test_balanced_accuracy"], color="#059669")
    ax.set_title("Best CV balanced accuracy by algorithm")
    ax.set_xlabel("Balanced accuracy")
    ax.set_xlim(0, max(1.0, frame["mean_test_balanced_accuracy"].max() + 0.05))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "model_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def number(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}f}"


def write_report(
    out_dir: Path,
    x: pd.DataFrame,
    y: pd.Series,
    labels: pd.DataFrame,
    description: str,
    feature_stats: pd.DataFrame,
    all_results: pd.DataFrame,
    searches: list[SearchResult],
    best_result: SearchResult,
    threshold: float,
    holdout_metrics: dict[str, Any],
    selected_features: list[str],
    surrogate_fidelity: float,
    surrogate_rules: str,
) -> Path:
    counts = y.value_counts().to_dict()
    pass_count = int(counts.get(0, 0))
    fail_count = int(counts.get(1, 0))
    total = len(y)

    total_missing = int(x.isna().sum().sum())
    total_cells = int(x.shape[0] * x.shape[1])
    features_with_missing = int((x.isna().sum(axis=0) > 0).sum())
    rows_with_missing = int((x.isna().sum(axis=1) > 0).sum())
    constant_features = int((x.nunique(dropna=True) <= 1).sum())
    duplicate_rows = int(x.duplicated().sum())

    top5 = feature_stats.dropna(subset=["cohen_abs"]).head(5)
    best_by_algorithm = (
        all_results.sort_values("mean_test_balanced_accuracy", ascending=False)
        .groupby("algorithm", as_index=False)
        .first()
        .sort_values("mean_test_balanced_accuracy", ascending=False)
    )

    hypothesis_rows = [
        (
            "Logistic Regression",
            "`select__k`: 20..120; `C`: 0.01..10; `penalty`: L1/L2",
            "Интерпретируемая линейная baseline-модель для большой размерности после стандартизации.",
        ),
        (
            "Gaussian Naive Bayes",
            "`select__k`: 10..80; `var_smoothing`: 1e-9..1e-3",
            "Быстрая вероятностная модель, часто устойчивая на малых выборках после отбора признаков.",
        ),
        (
            "KNN",
            "`select__k`: 10..40; `n_neighbors`: 3..25; `weights`: uniform/distance",
            "Нелинейная гипотеза локального сходства объектов после нормализации.",
        ),
        (
            "Decision Tree",
            "`select__k`: 10..80; `max_depth`: 2..5; `min_samples_leaf`: 10..40",
            "Пороговые правила удобны для диагностической интерпретации технологических состояний.",
        ),
        (
            "Random Forest",
            "`select__k`: 40..80; `max_depth`: 3/5/None; `min_samples_leaf`: 5/20",
            "Ансамбль деревьев проверяет устойчивые нелинейные зависимости и взаимодействия признаков.",
        ),
    ]

    report = f"""# SECOM: анализ, подготовка данных и построение модели

## 1. Анализ предметной области

### Априорные знания и ограничения

SECOM описывает процесс производства полупроводников: каждая запись соответствует одной производственной единице, а признаки являются измерениями сенсоров или контрольных точек процесса. Краткое описание исходного набора: {description}

Априорные знания:

- целевая метка бинарная: `-1` означает Pass, `1` означает Fail;
- Fail является редким, но критичным событием: {fail_count} из {total} объектов ({percent(fail_count / total)});
- признаки анонимизированы, поэтому модель выявляет статистические зависимости, а не автоматически доказанные физические причины брака;
- сенсорные признаки имеют разные масштабы и требуют нормализации;
- пропуски являются частью реальных технологических данных и должны обрабатываться внутри обучающего pipeline.

Ограничения:

- малый размер выборки при большой размерности: {x.shape[0]} объектов и {x.shape[1]} исходных признаков;
- сильный дисбаланс классов;
- возможные шум, корреляции и технологический дрейф;
- без расшифровки сенсоров интерпретация требует участия инженеров-технологов.

### Цель анализа, критерии качества и процедура проверки

Цель - построить модель раннего выявления риска `Fail` и выделить признаки, наиболее связанные со снижением выхода годных изделий.

Основной критерий качества - Balanced Accuracy, эквивалентно минимизации Balanced Error Rate: `BER = 1 - Balanced Accuracy`. Дополнительные критерии: Recall Fail, F1 Fail, ROC AUC. Обычная accuracy не выбрана основной, потому что при доле Fail {percent(fail_count / total, 1)} тривиальная модель почти всегда предсказывает Pass.

Процедура проверки:

- стратифицированное разделение на train/hold-out 80/20;
- на train-части - стратифицированная 5-fold cross-validation;
- медианная импутация, удаление признаков с большой долей пропусков, стандартизация и отбор признаков пересчитываются внутри каждого fold;
- лучшая гипотеза выбирается по `mean_test_balanced_accuracy`;
- после выбора модели threshold дополнительно настраивается на train через cross-validation;
- итоговая оценка выполняется один раз на hold-out.

### Пользовательские сценарии

1. Инженер-технолог загружает новые сенсорные измерения, система применяет сохраненный pipeline и выдает вероятность `Fail`.
2. Аналитик качества запускает переобучение и получает отчет с BER, Recall Fail и списком важных сенсоров.
3. Оператор мониторинга сортирует производственные единицы по риску `Fail` и отправляет верхнюю группу на дополнительную проверку.
4. Инженер процесса изучает top-признаки и суррогатное дерево для поиска технологических факторов.
5. Ответственный за ML-мониторинг отслеживает долю пропусков, распределения z-score и падение качества на новых данных.

## 2. Формирование и подготовка данных

### Сырые данные

В анализ включены все {x.shape[0]} строк из `secom.data` и соответствующие метки из `secom_labels.data`. Используются все исходные {x.shape[1]} сенсорных признаков, потому что признаки анонимизированы и заранее исключать технологические точки без предметной расшифровки рискованно. Временная метка сохранена для контроля периода наблюдений: {labels["timestamp"].min()} - {labels["timestamp"].max()}, но не используется как входной признак модели.

### Предобработка

- Проверка консистентности: число строк признаков совпадает с числом меток; дубликатов строк: {duplicate_rows}.
- Пропуски: признаки с долей пропусков выше 50% удаляются внутри обучающего fold, остальные значения заполняются медианой.
- Артефакты: константные признаки удаляются через `VarianceThreshold`.
- Нормализация: применяется `StandardScaler`, так как признаки имеют разные масштабы.
- Дискретизация: не используется как обязательный шаг, чтобы не терять информацию; пороговые правила строятся отдельно суррогатным деревом.

Сводка пропусков: всего пропущенных значений {total_missing} ({percent(total_missing / total_cells)}), признаки с пропусками: {features_with_missing}, строки с пропусками: {rows_with_missing}. Константных признаков: {constant_features}.

### Feature engineering

В pipeline реализованы:

- `missing_count` и `missing_rate` для каждой строки;
- `mean_abs_z` и `max_abs_z` как интегральные показатели отклонения объекта от типичного технологического состояния;
- фильтрация по доле пропусков;
- supervised feature selection через `SelectKBest(f_classif)`;
- перебор `select__k` как гиперпараметра.

PCA можно использовать как дополнительную визуализацию, но в целевой модели выбран отбор top-K признаков, потому что он лучше сохраняет интерпретируемость анонимизированных сенсоров.

### Разведочный анализ данных

Top-5 признаков по абсолютному Cohen's d:

| Признак | Доля пропусков | Mean Pass | Mean Fail | Cohen's d |
|---|---:|---:|---:|---:|
{chr(10).join(f"| {row.feature} | {percent(row.missing_rate)} | {number(row.pass_mean)} | {number(row.fail_mean)} | {number(row.cohen_abs)} |" for row in top5.itertuples())}

Графики:

![Target distribution](target_distribution.png)

![Missingness](missingness_top20.png)

![Top feature importance](top_feature_importance.png)

![Target vs top features](target_vs_top_features.png)

## 3. Построение модели и валидация

### Тип задачи

Это задача бинарной классификации с сильным дисбалансом классов. Модель оценивает вероятность класса `Fail`; дополнительно решается задача отбора значимых признаков для диагностики процесса.

### Гипотезы

| Гипотеза | Диапазон гиперпараметров | Обоснование |
|---|---|---|
{chr(10).join(f"| {name} | {params} | {why} |" for name, params, why in hypothesis_rows)}

### Результаты перебора

Лучшие модели по алгоритмам:

| Алгоритм | CV Balanced Accuracy | CV BER | CV Recall Fail | CV F1 Fail | Параметры |
|---|---:|---:|---:|---:|---|
{chr(10).join(f"| {row.algorithm} | {number(row.mean_test_balanced_accuracy)} | {number(1.0 - row.mean_test_balanced_accuracy)} | {number(row.mean_test_recall_fail)} | {number(row.mean_test_f1_fail)} | `{row.params_readable}` |" for row in best_by_algorithm.itertuples())}

Выбранная модель: **{best_result.algorithm}**.

Лучшие параметры: `{json.dumps(best_result.best_params, ensure_ascii=False)}`.

Подобранный threshold для класса Fail: `{threshold:.2f}`.

Hold-out метрики:

| Метрика | Значение |
|---|---:|
| Accuracy | {number(holdout_metrics["accuracy"])} |
| Balanced Accuracy | {number(holdout_metrics["balanced_accuracy"])} |
| BER | {number(holdout_metrics["ber"])} |
| Recall Fail | {number(holdout_metrics["recall_fail"])} |
| Recall Pass | {number(holdout_metrics["recall_pass"])} |
| Precision Fail | {number(holdout_metrics["precision_fail"])} |
| F1 Fail | {number(holdout_metrics["f1_fail"])} |
| ROC AUC | {number(holdout_metrics["roc_auc"])} |
| TP / FN / FP / TN | {holdout_metrics["tp"]} / {holdout_metrics["fn"]} / {holdout_metrics["fp"]} / {holdout_metrics["tn"]} |

![Model comparison](model_comparison.png)

## 4. Интерпретация и объяснение модели

Построена глобальная суррогатная модель - дерево решений глубины 3, обученное воспроизводить прогнозы выбранной модели на подготовленных признаках. Fidelity суррогата: {number(surrogate_fidelity)}.

Правила сохранены в `surrogate_rules.txt`, изображение дерева - `surrogate_tree.png`.

Первые выбранные признаки итогового pipeline:

{", ".join(selected_features[:20])}

Фрагмент правил:

```text
{chr(10).join(surrogate_rules.splitlines()[:18])}
```

## Вывод

SECOM следует решать как дисбалансную бинарную классификацию с фокусом на BER/Balanced Accuracy и Recall Fail. Для практического применения важно сохранять весь pipeline предобработки, регулярно проверять дрейф распределений и рассматривать найденные top-признаки как кандидаты для технологического расследования.
"""

    report_path = out_dir / "secom_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SECOM coursework analysis")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to secom.data")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH, help="Path to secom_labels.data")
    parser.add_argument("--names", type=Path, default=DEFAULT_NAMES_PATH, help="Path to secom.names")
    parser.add_argument("--out-dir", type=Path, default=Path("secom_python_outputs"), help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for sklearn")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    print("Loading data...")
    x, y, labels = load_secom(args.data, args.labels)
    description = read_dataset_description(args.names)

    print("Running EDA...")
    feature_stats = compute_feature_statistics(x, y)
    feature_stats.to_csv(out_dir / "feature_statistics.csv", index=False)
    save_target_distribution_plot(y, out_dir)
    save_missingness_plot(feature_stats, out_dir)
    save_feature_importance_plot(feature_stats, out_dir)
    save_target_vs_features_plot(x, y, feature_stats, out_dir)

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("Running model selection...")
    searches, all_results = run_model_selection(x_train, y_train, n_jobs=args.n_jobs)
    all_results.to_csv(out_dir / "model_selection_results.csv", index=False)
    best_result = searches[0]

    best_by_algorithm = (
        all_results.sort_values("mean_test_balanced_accuracy", ascending=False)
        .groupby("algorithm", as_index=False)
        .first()
        .sort_values("mean_test_balanced_accuracy", ascending=False)
    )
    save_model_comparison_plot(best_by_algorithm, out_dir)

    print("Choosing threshold...")
    threshold, threshold_results = choose_threshold_by_cv(best_result.best_estimator, x_train, y_train, n_jobs=args.n_jobs)
    threshold_results.to_csv(out_dir / "threshold_selection.csv", index=False)

    print("Evaluating hold-out...")
    holdout_metrics = evaluate_at_threshold(best_result.best_estimator, x_holdout, y_holdout, threshold)
    (out_dir / "holdout_classification_report.txt").write_text(holdout_metrics["classification_report"], encoding="utf-8")

    feature_names = selected_feature_names(best_result.best_estimator, list(x.columns))
    pd.DataFrame({"selected_feature": feature_names}).to_csv(out_dir / "selected_features.csv", index=False)

    print("Building surrogate tree...")
    _, surrogate_fidelity, surrogate_rules, _ = build_surrogate(
        best_estimator=best_result.best_estimator,
        threshold=threshold,
        x_train=x_train,
        y_train=y_train,
        feature_names=feature_names,
        out_dir=out_dir,
    )

    print("Saving model and report...")
    joblib.dump(
        {
            "model": best_result.best_estimator,
            "threshold": threshold,
            "input_columns": list(x.columns),
            "target_encoding": {"Pass": 0, "Fail": 1},
        },
        out_dir / "best_secom_model.joblib",
    )

    summary = {
        "best_algorithm": best_result.algorithm,
        "best_params": best_result.best_params,
        "cv_best_balanced_accuracy": best_result.best_score,
        "threshold": threshold,
        "holdout_metrics": {k: v for k, v in holdout_metrics.items() if k != "classification_report"},
        "surrogate_fidelity": surrogate_fidelity,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = write_report(
        out_dir=out_dir,
        x=x,
        y=y,
        labels=labels,
        description=description,
        feature_stats=feature_stats,
        all_results=all_results,
        searches=searches,
        best_result=best_result,
        threshold=threshold,
        holdout_metrics=holdout_metrics,
        selected_features=feature_names,
        surrogate_fidelity=surrogate_fidelity,
        surrogate_rules=surrogate_rules,
    )
    print(f"Done. Report: {report_path.resolve()}")
    print(f"Artifacts: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
