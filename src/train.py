from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data set.csv"
MODEL_DIR = PROJECT_ROOT / "models"
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test"
TEST_RAW_OUTPUT_PATH = TEST_DATA_DIR / "test_raw_15_percent.csv"

ID_COL = "CONS_NO"
TARGET_COL = "FLAG"
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


def log(message: str) -> None:
    print(f"[TRAIN] {message}")


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_feature_table(features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    if not features_path.exists():
        raise FileNotFoundError(f"Feature table not found: {features_path}")

    df = pd.read_csv(features_path)
    required_cols = {ID_COL, TARGET_COL}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    feature_cols = [col for col in df.columns if col not in [ID_COL, TARGET_COL]]
    X_all = df[feature_cols].copy()
    y_all = df[TARGET_COL].astype(int).copy()
    ids_all = df[ID_COL].copy()

    n_missing = int(X_all.isna().sum().sum())
    n_inf = int(np.isinf(X_all.to_numpy(dtype=np.float64)).sum())
    if n_missing or n_inf:
        raise ValueError(f"Feature matrix is not clean: missing={n_missing}, inf={n_inf}")

    return df, X_all, y_all, ids_all, feature_cols


def split_data(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    ids_all: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    temp_size = VALIDATION_SIZE + TEST_SIZE
    relative_test_size = TEST_SIZE / temp_size

    X_train_raw, X_temp_raw, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X_all,
        y_all,
        ids_all,
        test_size=temp_size,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )
    X_val_raw, X_test_raw, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp_raw,
        y_temp,
        ids_temp,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return (
        X_train_raw.reset_index(drop=True),
        X_val_raw.reset_index(drop=True),
        X_test_raw.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
        ids_train.reset_index(drop=True),
        ids_val.reset_index(drop=True),
        ids_test.reset_index(drop=True),
    )


def prepare_features(
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    constant_cols = [col for col in X_train_raw.columns if X_train_raw[col].nunique(dropna=False) <= 1]
    active_feature_cols = [col for col in X_train_raw.columns if col not in constant_cols]

    return (
        X_train_raw[active_feature_cols].copy(),
        X_val_raw[active_feature_cols].copy(),
        X_test_raw[active_feature_cols].copy(),
        active_feature_cols,
        constant_cols,
    )


def safe_score(estimator: Any, X: pd.DataFrame, score_kind: str = "predict_proba") -> np.ndarray:
    if score_kind == "predict_proba":
        return estimator.predict_proba(X)[:, 1]
    raise ValueError(f"Unsupported score_kind: {score_kind}")


def metrics_at_threshold(y_true: pd.Series, score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score),
        "pr_auc": average_precision_score(y_true, score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_grid(score: np.ndarray) -> np.ndarray:
    return np.unique(np.r_[np.linspace(0, 1, 501), np.quantile(score, np.linspace(0, 1, 501))])


def tune_thresholds(y_true: pd.Series, score: np.ndarray) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    curve = pd.DataFrame([metrics_at_threshold(y_true, score, t) for t in threshold_grid(score)])
    choices = {
        "default_0_50": metrics_at_threshold(y_true, score, 0.50),
        "best_f1": curve.loc[curve["f1"].idxmax()].to_dict(),
        "best_f2": curve.loc[curve["f2"].idxmax()].to_dict(),
    }
    for target_recall in [0.65, 0.80]:
        eligible = curve[curve["recall"] >= target_recall]
        key = f"recall_at_least_{str(target_recall).replace('.', '_')}"
        choices[key] = (
            eligible.loc[eligible["f1"].idxmax()].to_dict()
            if len(eligible)
            else curve.loc[curve["recall"].idxmax()].to_dict()
        )
    return choices, curve


def compact_metrics(row: dict[str, Any]) -> dict[str, Any]:
    keys = ["threshold", "accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]
    return {key: row[key] for key in keys}


def build_model_specs(y_train: pd.Series) -> tuple[list[dict[str, Any]], float]:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("lightgbm is required because the web demo uses the LightGBM inference bundle.") from exc

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model_specs: list[dict[str, Any]] = [
        {
            "key": "logistic_l2_balanced",
            "display_name": "Logistic Regression + L2 + balanced",
            "group": "Linear baseline",
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="l2",
                            C=1.0,
                            solver="lbfgs",
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "score_kind": "predict_proba",
        },
        {
            "key": "random_forest",
            "display_name": "Random Forest benchmark",
            "group": "Bagging",
            "estimator": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "score_kind": "predict_proba",
        },
        {
            "key": "lightgbm",
            "display_name": "LightGBM benchmark",
            "group": "Boosting",
            "estimator": LGBMClassifier(
                n_estimators=450,
                learning_rate=0.035,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=80,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            "score_kind": "predict_proba",
        },
    ]

    return model_specs, scale_pos_weight


def train_candidates(
    model_specs: list[dict[str, Any]],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    model_results: dict[str, dict[str, Any]] = {}
    training_rows = []

    for spec in model_specs:
        log(f"Training {spec['display_name']}")
        started = time.perf_counter()
        estimator = clone(spec["estimator"])
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        val_score = safe_score(estimator, X_val, spec["score_kind"])
        test_score = safe_score(estimator, X_test, spec["score_kind"])
        threshold_choices, threshold_curve = tune_thresholds(y_val, val_score)

        model_results[spec["key"]] = {
            **spec,
            "estimator": estimator,
            "fit_seconds": fit_seconds,
            "val_score": val_score,
            "test_score": test_score,
            "threshold_choices": threshold_choices,
            "threshold_curve": threshold_curve,
        }

        training_rows.append(
            {
                "model_key": spec["key"],
                "Model": spec["display_name"],
                "Group": spec["group"],
                "Fit seconds": fit_seconds,
                "Val ROC-AUC": threshold_choices["default_0_50"]["roc_auc"],
                "Val PR-AUC": threshold_choices["default_0_50"]["pr_auc"],
                "Val F1 @0.50": threshold_choices["default_0_50"]["f1"],
                "Val Recall @0.50": threshold_choices["default_0_50"]["recall"],
                "Best F1": threshold_choices["best_f1"]["f1"],
                "Best F1 threshold": threshold_choices["best_f1"]["threshold"],
                "Best F2": threshold_choices["best_f2"]["f2"],
                "Best F2 threshold": threshold_choices["best_f2"]["threshold"],
            }
        )

    return model_results, pd.DataFrame(training_rows)


def build_threshold_report(model_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for key, result in model_results.items():
        for threshold_name, metrics in result["threshold_choices"].items():
            rows.append(
                {
                    "model_key": key,
                    "Model": result["display_name"],
                    "Group": result["group"],
                    "Threshold policy": threshold_name,
                    **compact_metrics(metrics),
                }
            )
    return pd.DataFrame(rows)


def evaluate_on_test(
    model_results: dict[str, dict[str, Any]],
    y_test: pd.Series,
) -> pd.DataFrame:
    rows = []
    for key, result in model_results.items():
        threshold = float(result["threshold_choices"]["best_f2"]["threshold"])
        metrics = metrics_at_threshold(y_test, result["test_score"], threshold)
        rows.append(
            {
                "model_key": key,
                "Model": result["display_name"],
                "Group": result["group"],
                "Threshold": threshold,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "F2": metrics["f2"],
                "PR-AUC": metrics["pr_auc"],
                "ROC-AUC": metrics["roc_auc"],
                "FP": metrics["fp"],
                "FN": metrics["fn"],
                "TP": metrics["tp"],
            }
        )
    return pd.DataFrame(rows).sort_values(["F2", "PR-AUC"], ascending=False).reset_index(drop=True)


def export_test_raw_csv(raw_data_path: Path, output_path: Path, ids_test: pd.Series) -> dict[str, Any]:
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_data_path}")

    raw_df = pd.read_csv(raw_data_path)
    if ID_COL not in raw_df.columns:
        raise ValueError(f"Raw data must contain {ID_COL}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_ids = raw_df[ID_COL].astype(str)
    ids_order = ids_test.astype(str).tolist()
    missing_ids = sorted(set(ids_order) - set(raw_ids))
    if missing_ids:
        raise ValueError(f"Cannot export test raw split; missing {len(missing_ids)} IDs in raw data.")

    test_raw = raw_df.assign(**{ID_COL: raw_ids}).set_index(ID_COL).loc[ids_order].reset_index()
    test_raw = test_raw[raw_df.columns]
    test_raw.to_csv(output_path, index=False)

    return {
        "path": str(output_path.relative_to(PROJECT_ROOT)),
        "rows": len(test_raw),
        "columns": len(test_raw.columns),
        "theft_ratio": float(test_raw[TARGET_COL].mean()) if TARGET_COL in test_raw.columns else None,
    }


def save_outputs(
    output_dir: Path,
    model_results: dict[str, dict[str, Any]],
    training_summary: pd.DataFrame,
    threshold_report: pd.DataFrame,
    test_comparison: pd.DataFrame,
    active_feature_cols: list[str],
    constant_cols: list[str],
    scale_pos_weight: float,
    split_report: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_row = test_comparison.iloc[0].to_dict()
    best_key = best_row["model_key"]
    if "lightgbm" not in model_results:
        raise ValueError("LightGBM result is required to save the inference bundle.")

    lightgbm_result = model_results["lightgbm"]
    lightgbm_test_row = test_comparison[test_comparison["model_key"] == "lightgbm"].iloc[0].to_dict()
    lightgbm_threshold = float(lightgbm_result["threshold_choices"]["best_f2"]["threshold"])
    bundle = {
        "model_key": "lightgbm",
        "model_name": lightgbm_result["display_name"],
        "model_group": lightgbm_result["group"],
        "estimator": lightgbm_result["estimator"],
        "score_kind": lightgbm_result["score_kind"],
        "threshold": lightgbm_threshold,
        "threshold_policy": "best_f2_on_validation",
        "active_feature_cols": active_feature_cols,
        "constant_cols": constant_cols,
        "id_col": ID_COL,
        "target_col": TARGET_COL,
        "random_state": RANDOM_STATE,
        "test_metrics": lightgbm_test_row,
    }

    with (output_dir / "energy_theft_model_bundle.pkl").open("wb") as f:
        pickle.dump(bundle, f)

    training_summary.to_csv(output_dir / "training_summary.csv", index=False)
    threshold_report.to_csv(output_dir / "threshold_report.csv", index=False)
    test_comparison.to_csv(output_dir / "test_comparison.csv", index=False)

    metadata = {
        "best_model_key": best_key,
        "best_model_name": best_row["Model"],
        "best_threshold": best_row["Threshold"],
        "inference_bundle": {
            "path": str((output_dir / "energy_theft_model_bundle.pkl").relative_to(PROJECT_ROOT)),
            "model_key": "lightgbm",
            "model_name": lightgbm_result["display_name"],
            "threshold": lightgbm_threshold,
            "contains_only_inference_model": True,
        },
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "split": split_report,
        "active_feature_count": len(active_feature_cols),
        "constant_features_dropped": constant_cols,
        "metrics": {
            "training_summary": training_summary.to_dict(orient="records"),
            "test_comparison": test_comparison.to_dict(orient="records"),
        },
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(to_builtin(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Energy Theft Detection models.")
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    parser.add_argument("--raw-data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--test-raw-output", type=Path, default=TEST_RAW_OUTPUT_PATH)
    args = parser.parse_args()

    log(f"Loading features from {args.features_path}")
    df, X_all, y_all, ids_all, feature_cols = load_feature_table(args.features_path)
    split_data_tuple = split_data(X_all, y_all, ids_all)
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, ids_train, ids_val, ids_test = split_data_tuple
    X_train, X_val, X_test, active_feature_cols, constant_cols = prepare_features(X_train_raw, X_val_raw, X_test_raw)

    assert len(X_train) + len(X_val) + len(X_test) == len(X_all)
    assert ids_train.isin(ids_val).sum() == 0
    assert ids_train.isin(ids_test).sum() == 0
    assert ids_val.isin(ids_test).sum() == 0

    model_specs, scale_pos_weight = build_model_specs(y_train)
    model_results, training_summary = train_candidates(model_specs, X_train, X_val, X_test, y_train, y_val)
    threshold_report = build_threshold_report(model_results)
    test_comparison = evaluate_on_test(model_results, y_test)
    test_raw_export = export_test_raw_csv(args.raw_data_path, args.test_raw_output, ids_test)

    split_report = {
        "full_rows": len(y_all),
        "train_rows": len(y_train),
        "validation_rows": len(y_val),
        "test_rows": len(y_test),
        "full_theft_ratio": float(y_all.mean()),
        "train_theft_ratio": float(y_train.mean()),
        "validation_theft_ratio": float(y_val.mean()),
        "test_theft_ratio": float(y_test.mean()),
        "raw_feature_count": len(feature_cols),
        "active_feature_count": len(active_feature_cols),
        "test_raw_export": test_raw_export,
    }

    save_outputs(
        output_dir=args.output_dir,
        model_results=model_results,
        training_summary=training_summary,
        threshold_report=threshold_report,
        test_comparison=test_comparison,
        active_feature_cols=active_feature_cols,
        constant_cols=constant_cols,
        scale_pos_weight=scale_pos_weight,
        split_report=split_report,
    )

    log("Training summary")
    print(training_summary.round(4).to_string(index=False))
    log("Test comparison at validation Best F2 thresholds")
    print(test_comparison.round(4).to_string(index=False))
    log(f"Saved artifacts to {args.output_dir}")
    log(f"Saved exact raw test split to {args.test_raw_output}")


if __name__ == "__main__":
    main()
