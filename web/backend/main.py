"""
FastAPI backend for Energy Theft Detection web demo.
Loads the LightGBM model bundle and exposes prediction endpoints.
"""
from __future__ import annotations

import io
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference_pipeline import predict_from_raw_csv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUNDLE_PATH = PROJECT_ROOT / "models" / "energy_theft_model_bundle.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
SAMPLE_CSV_PATH = PROJECT_ROOT / "data" / "test" / "test_raw_15_percent.csv"

# ---------------------------------------------------------------------------
# Load model bundle at startup
# ---------------------------------------------------------------------------
if not BUNDLE_PATH.exists():
    raise FileNotFoundError(f"Model bundle not found: {BUNDLE_PATH}")

with BUNDLE_PATH.open("rb") as f:
    MODEL_BUNDLE: dict = pickle.load(f)

MODEL_METADATA: dict = {}
if METADATA_PATH.exists():
    MODEL_METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Energy Theft Detection API",
    description="Web demo API for SGCC electricity theft detection using LightGBM.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/model-info")
def model_info():
    threshold = MODEL_BUNDLE.get("threshold", 0)
    active_features = MODEL_BUNDLE.get("active_feature_cols", [])

    # Get test metrics from metadata
    test_metrics = {}
    test_comparison = MODEL_METADATA.get("metrics", {}).get("test_comparison", [])
    for entry in test_comparison:
        if entry.get("model_key") == "lightgbm":
            test_metrics = entry
            break

    return {
        "model_name": MODEL_BUNDLE.get("model_name", "LightGBM benchmark"),
        "model_key": MODEL_BUNDLE.get("model_key", "lightgbm"),
        "threshold": round(threshold, 6),
        "threshold_policy": MODEL_BUNDLE.get("threshold_policy", "best_f2_on_validation"),
        "feature_count": len(active_features),
        "test_metrics": {
            "F2": round(test_metrics.get("F2", 0), 4),
            "Recall": round(test_metrics.get("Recall", 0), 4),
            "Precision": round(test_metrics.get("Precision", 0), 4),
            "PR_AUC": round(test_metrics.get("PR-AUC", 0), 4),
            "ROC_AUC": round(test_metrics.get("ROC-AUC", 0), 4),
            "F1": round(test_metrics.get("F1", 0), 4),
        },
        "split": MODEL_METADATA.get("split", {}),
    }


@app.get("/sample-info")
def sample_info():
    if not SAMPLE_CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample CSV not found.")

    # Read just first few rows for shape info
    df = pd.read_csv(SAMPLE_CSV_PATH, nrows=5)
    full_shape = pd.read_csv(SAMPLE_CSV_PATH, usecols=[0]).shape[0]

    return {
        "path": str(SAMPLE_CSV_PATH.relative_to(PROJECT_ROOT)),
        "rows": full_shape,
        "columns": len(df.columns),
        "available": True,
    }


@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV is empty.")

    start_time = time.time()

    try:
        records = predict_from_raw_csv(df, MODEL_BUNDLE)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    elapsed = time.time() - start_time
    threshold = MODEL_BUNDLE.get("threshold", 0)

    scores = [r["score"] for r in records]
    predicted_theft = sum(1 for r in records if r["prediction"] == "Suspected theft")
    predicted_normal = len(records) - predicted_theft

    # Confusion summary if ground truth available
    confusion = None
    has_flag = any("actual_label" in r for r in records)
    if has_flag:
        tp = sum(1 for r in records if r.get("outcome") == "TP")
        tn = sum(1 for r in records if r.get("outcome") == "TN")
        fp = sum(1 for r in records if r.get("outcome") == "FP")
        fn = sum(1 for r in records if r.get("outcome") == "FN")
        confusion = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    return {
        "model": MODEL_BUNDLE.get("model_name", "LightGBM benchmark"),
        "threshold": round(threshold, 6),
        "rows": len(records),
        "elapsed_seconds": round(elapsed, 2),
        "summary": {
            "predicted_theft": predicted_theft,
            "predicted_normal": predicted_normal,
            "average_score": round(float(np.mean(scores)), 4),
        },
        "confusion": confusion,
        "records": records,
    }


@app.post("/predict/sample")
async def predict_sample():
    """Run prediction on the built-in test sample CSV."""
    if not SAMPLE_CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample CSV not found on server.")

    df = pd.read_csv(SAMPLE_CSV_PATH)

    start_time = time.time()

    try:
        records = predict_from_raw_csv(df, MODEL_BUNDLE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    elapsed = time.time() - start_time
    threshold = MODEL_BUNDLE.get("threshold", 0)

    scores = [r["score"] for r in records]
    predicted_theft = sum(1 for r in records if r["prediction"] == "Suspected theft")
    predicted_normal = len(records) - predicted_theft

    confusion = None
    has_flag = any("actual_label" in r for r in records)
    if has_flag:
        tp = sum(1 for r in records if r.get("outcome") == "TP")
        tn = sum(1 for r in records if r.get("outcome") == "TN")
        fp = sum(1 for r in records if r.get("outcome") == "FP")
        fn = sum(1 for r in records if r.get("outcome") == "FN")
        confusion = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    return {
        "model": MODEL_BUNDLE.get("model_name", "LightGBM benchmark"),
        "threshold": round(threshold, 6),
        "rows": len(records),
        "elapsed_seconds": round(elapsed, 2),
        "summary": {
            "predicted_theft": predicted_theft,
            "predicted_normal": predicted_normal,
            "average_score": round(float(np.mean(scores)), 4),
        },
        "confusion": confusion,
        "records": records,
    }
