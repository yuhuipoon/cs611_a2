import os
import re
import json
import argparse

import joblib
import pandas as pd
import pyspark
from pyspark.sql.functions import col

META_COLS = ["Customer_ID", "loan_id", "snapshot_date", "label_date", "label", "label_def"]
EMA_COLS  = [f"ema_fe_{i}" for i in range(1, 21)]
REGISTRY  = "model_registry/registry.json"


def load_registry():
    if not os.path.exists(REGISTRY):
        raise FileNotFoundError(f"[inference] registry not found: {REGISTRY}")
    with open(REGISTRY) as f:
        return json.load(f)


def load_model(version):
    path = os.path.join("model_registry", f"{version}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[inference] model not found: {path}")
    return joblib.load(path)


def load_feature_store(gold_features_dir, snapshot_date_str, spark):
    path = os.path.join(gold_features_dir, "gold_features.parquet")
    df = spark.read.parquet(path).filter(col("snapshot_date") == snapshot_date_str)
    print(f"[inference] {snapshot_date_str} feature rows: {df.count()}")
    return df


def prepare_features(df, artefact):
    pdf = df.toPandas()

    # scale EMA columns using the fitted scaler from the artefact
    scaler = artefact["preprocessing_transformers"]["stdscaler"]
    pdf[EMA_COLS] = scaler.transform(pdf[EMA_COLS])

    feature_cols = artefact["feature_cols"]
    X = pdf[[c for c in feature_cols if c in pdf.columns]].copy()

    # one-hot encode string columns and align to training columns
    str_cols = [c for c in X.columns if X[c].dtype == object]
    if str_cols:
        X = pd.get_dummies(X, columns=str_cols, drop_first=False)
    # sanitize names identically to training (XGBoost rejects [ ] < ; pd.cut
    # bins produce such names). Must match get_feature_matrix or the reindex
    # below would drop every encoded column and zero out the features.
    X.columns = [re.sub(r"[\[\]<>(),]", "_", str(c)) for c in X.columns]
    X = X.reindex(columns=feature_cols, fill_value=0)
    # impute NaN EMA (no-clickstream customers) to 0 — matches training
    X = X.fillna(0)

    return pdf, X


def run_inference(snapshot_date_str, gold_features_dir, output_dir, spark):
    # first DAG run trains the production model but has nothing to score yet —
    # check this BEFORE touching the registry (training may not have created it)
    if snapshot_date_str == "2024-09-01":
        print(f"[inference] first DAG run — no model deployed yet, skipping inference.")
        return

    if not os.path.exists(REGISTRY):
        print(f"[inference] registry not found yet ({REGISTRY}), skipping.")
        return

    registry = load_registry()
    if not registry.get("production"):
        print(f"[inference] no production model in registry yet, skipping.")
        return

    df = load_feature_store(gold_features_dir, snapshot_date_str, spark)

    # skip when there is no feature data for this snapshot month — past the feature
    # horizon (feature sources end ~2024-12/2025-01). Nothing to score.
    if df.count() == 0:
        print(f"[inference] no feature data for {snapshot_date_str} (past feature horizon), skipping.")
        return

    # collect models to score: production + the autoML trained in the previous month
    models_to_run = [("production", registry["production"])]
    from dateutil.relativedelta import relativedelta
    from datetime import datetime
    prev_month_str = (datetime.strptime(snapshot_date_str, "%Y-%m-%d") - relativedelta(months=1)).strftime("%Y-%m-%d")
    prev_version = f"credit_model_{prev_month_str.replace('-', '_')}"
    automl_list = registry.get("autoML", [])
    matched = [e for e in automl_list if e["version"] == prev_version]
    if matched:
        models_to_run.append((matched[0]["name"], matched[0]["version"]))
        print(f"[inference] autoML model: {matched[0]['name']} ({matched[0]['version']})")
    else:
        print(f"[inference] no autoML model found for previous month ({prev_version}), running production only.")

    batch_dir = os.path.join(output_dir, f"batch_{snapshot_date_str}")
    os.makedirs(batch_dir, exist_ok=True)

    for model_label, version in models_to_run:
        artefact = load_model(version)
        pdf, X = prepare_features(df, artefact)

        proba = artefact["model"].predict_proba(X)[:, 1] # calibrator is here
        threshold = artefact.get("decision_threshold", 0.5)
        pdf["score"]          = proba
        pdf["prediction"]     = (proba >= threshold).astype(int)  # class at the model's tuned threshold
        pdf["threshold"]      = threshold
        pdf["model_version"]  = version
        pdf["model_label"]    = model_label
        pdf["inference_date"] = snapshot_date_str

        top10 = artefact.get("top10_features", [])
        top10_present = [f for f in top10 if f in pdf.columns]
        out_cols = ["Customer_ID", "score", "prediction", "threshold",
                    "model_version", "model_label", "inference_date"] + top10_present
        out_path = os.path.join(batch_dir, f"{model_label}.parquet")
        pdf[out_cols].to_parquet(out_path, index=False)
        print(f"[inference] {model_label} ({version}) | rows: {len(pdf)} | mean score: {proba.mean():.4f} | saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("inference") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    run_inference(
        args.snapshotdate,
        "datamart/gold/features/",
        "datamart/gold/inference/",
        spark,
    )

    spark.stop()
