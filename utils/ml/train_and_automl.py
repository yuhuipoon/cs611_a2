import os
import argparse

import pyspark
from pyspark.sql.functions import col

# columns that are metadata, not features
META_COLS = ["Customer_ID", "loan_id", "snapshot_date", "label_date", "label", "label_def"]


def load_feature_store(gold_features_dir, spark):
    path = os.path.join(gold_features_dir, "gold_features.parquet")
    df = spark.read.parquet(path)
    print(f"[train] feature store loaded | rows: {df.count()} | cols: {len(df.columns)}")
    return df


def load_label_store(gold_label_dir, spark):
    path = os.path.join(gold_label_dir, "gold_label_store.parquet")
    df = spark.read.parquet(path)
    print(f"[train] label store loaded | rows: {df.count()} | label distribution:")
    df.groupBy("label").count().show()
    return df


def build_training_data(df_features, df_labels):
    # rename label snapshot_date to avoid conflict with feature snapshot_date
    df_labels = df_labels.withColumnRenamed("snapshot_date", "label_date")

    # each customer has exactly one loan -> one row in each table -> join on Customer_ID alone
    df = df_labels.join(df_features, on="Customer_ID", how="left")

    print(f"[train] training set | rows: {df.count()} | cols: {len(df.columns)}")
    df.groupBy("label").count().show()
    return df


EMA_COLS = [f"ema_fe_{i}" for i in range(1, 21)]


def scale_ema_features(train, val, test, oot, model_dir, snapshot_date_str):
    from sklearn.preprocessing import StandardScaler
    import joblib

    scaler = StandardScaler()
    train[EMA_COLS] = scaler.fit_transform(train[EMA_COLS])
    val[EMA_COLS]   = scaler.transform(val[EMA_COLS])
    test[EMA_COLS]  = scaler.transform(test[EMA_COLS])
    oot[EMA_COLS]   = scaler.transform(oot[EMA_COLS])

    os.makedirs(model_dir, exist_ok=True)
    ds = snapshot_date_str.replace("-", "_")
    scaler_path = os.path.join(model_dir, f"scaler_{ds}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[train] scaler saved: {scaler_path}")

    return train, val, test, oot, scaler


def split_data(df):
    import pandas as pd

    pdf = df.toPandas()
    pdf["label_date"] = pd.to_datetime(pdf["label_date"])

    # OOT: last 2 months by label_date
    sorted_months = sorted(pdf["label_date"].unique())
    oot_months = sorted_months[-2:]
    print(f"[train] OOT months: {[str(m.date()) for m in oot_months]}")

    oot = pdf[pdf["label_date"].isin(oot_months)]
    remaining = pdf[~pdf["label_date"].isin(oot_months)]
    print(f"[train] OOT rows: {len(oot)} | remaining rows: {len(remaining)}")

    # 80/10/10 split on remaining
    from sklearn.model_selection import train_test_split
    remaining = remaining.sample(frac=1, random_state=42).reset_index(drop=True)
    train, temp = train_test_split(remaining, test_size=0.2, random_state=42, stratify=remaining["label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])

    print(f"[train] train: {len(train)} | val: {len(val)} | test: {len(test)} | oot: {len(oot)}")
    print(f"[train] label rate — train: {train['label'].mean():.3f} | val: {val['label'].mean():.3f} | test: {test['label'].mean():.3f} | oot: {oot['label'].mean():.3f}")

    return train, val, test, oot


def get_feature_matrix(df, feature_cols):
    """One-hot encode remaining string columns and return X, y as numpy arrays."""
    import re
    import pandas as pd
    X = df[feature_cols].copy()
    # encode any leftover string/object columns (binned cols)
    str_cols = [c for c in X.columns if X[c].dtype == object]
    if str_cols:
        X = pd.get_dummies(X, columns=str_cols, drop_first=False)
    # XGBoost rejects feature names containing [ ] < ; pd.cut bins like
    # "(0.999, 5.0]" produce such names. Sanitize here AND identically in
    # inference.prepare_features so the columns line up.
    X.columns = [re.sub(r"[\[\]<>(),]", "_", str(c)) for c in X.columns]
    # EMA features are NaN for customers with no clickstream history (clickstream
    # source ends 2024-12). Impute to 0 = the mean in StandardScaler space; the
    # has_clickstream flag already signals missingness. Required since LR rejects NaN.
    X = X.fillna(0)
    y = df["label"].values
    return X, y


def compute_metrics(y_true, y_prob, threshold=0.5):
    from sklearn.metrics import roc_auc_score, recall_score
    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "gini":   round(2 * auc - 1, 4),
        "recall": round(recall_score(y_true, y_pred), 4),
    }


def train_lr(train_X, train_y, val_X, val_y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV

    param_dist = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["saga"],
        "max_iter": [1000],
    }
    base = LogisticRegression(random_state=42, class_weight="balanced")
    search = RandomizedSearchCV(base, param_dist, n_iter=6, scoring="roc_auc",
                                cv=3, random_state=42, n_jobs=-1)
    search.fit(train_X, train_y)
    best = search.best_estimator_
    m = compute_metrics(val_y, best.predict_proba(val_X)[:, 1])
    print(f"[lr] best params: {search.best_params_} | val gini: {m['gini']} | val recall: {m['recall']}")
    return best, search.best_params_


def train_xgb(train_X, train_y, val_X, val_y):
    from xgboost import XGBClassifier
    from sklearn.model_selection import RandomizedSearchCV

    scale_pos_weight = (train_y == 0).sum() / (train_y == 1).sum()
    # heavily regularized: shallow trees + strong leaf/penalty/subsample controls to
    # close the train-vs-OOT gap. Depth is the main overfitting lever, kept low.
    param_dist = {
        "n_estimators": [300, 500],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.02],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "min_child_weight": [5, 10, 20],
        "reg_lambda": [5, 10, 20],
        "reg_alpha": [0, 0.5, 1],
        "gamma": [0.5, 1, 2],
    }
    base = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight,
                         use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
    search = RandomizedSearchCV(base, param_dist, n_iter=30, scoring="roc_auc",
                                cv=3, random_state=42, n_jobs=-1)
    search.fit(train_X, train_y)
    best = search.best_estimator_
    m = compute_metrics(val_y, best.predict_proba(val_X)[:, 1])
    print(f"[xgb] best params: {search.best_params_} | val gini: {m['gini']} | val recall: {m['recall']}")
    return best, search.best_params_


def calibrate(model, val_X, val_y):
    # Platt scaling (sigmoid): fits a logistic map from raw scores to calibrated
    # probabilities. More robust than isotonic on small validation sets.
    from sklearn.calibration import CalibratedClassifierCV
    cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    cal.fit(val_X, val_y)
    return cal


def select_threshold(model, val_X, val_y, min_precision=0.5):
    # maximise recall on the validation set subject to a minimum precision floor,
    # so the threshold isn't trivially pushed to 0 (flag everyone as default).
    import numpy as np
    from sklearn.metrics import recall_score, precision_score
    probs = model.predict_proba(val_X)[:, 1]
    best_t, best_recall = 0.5, -1.0
    for t in np.round(np.linspace(0.05, 0.95, 19), 3):
        preds = (probs >= t).astype(int)
        prec = precision_score(val_y, preds, zero_division=0)
        if prec < min_precision:
            continue
        rec = recall_score(val_y, preds, zero_division=0)
        if rec > best_recall:
            best_recall, best_t = rec, float(t)
    print(f"[threshold] tuned on val (max recall={best_recall:.3f}, min_precision={min_precision}): {best_t}")
    return best_t


def evaluate(name, model, splits, threshold=0.5):
    print(f"\n[{name}] evaluation (threshold={threshold}):")
    results = {}
    for split_name, X, y in splits:
        m = compute_metrics(y, model.predict_proba(X)[:, 1], threshold=threshold)
        print(f"  {split_name:<6} gini: {m['gini']} | recall: {m['recall']}")
        results[split_name] = m
    return results


def get_feature_importances(model, feature_cols, top_n=10):
    import numpy as np
    # unwrap CalibratedClassifierCV to get the base estimator
    base = model.estimator if hasattr(model, "estimator") else model.calibrated_classifiers_[0].estimator
    if hasattr(base, "feature_importances_"):
        importances = base.feature_importances_
    elif hasattr(base, "coef_"):
        importances = np.abs(base.coef_[0])
    else:
        return []
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:top_n]]


def train_models(train, val, test, oot, model_dir, scaler, snapshot_date_str):
    import joblib

    # derive feature columns: everything except metadata
    feature_cols = [c for c in train.columns if c not in META_COLS]

    train_X, train_y = get_feature_matrix(train, feature_cols)
    val_X,   val_y   = get_feature_matrix(val,   feature_cols)
    test_X,  test_y  = get_feature_matrix(test,  feature_cols)
    oot_X,   oot_y   = get_feature_matrix(oot,   feature_cols)

    # align columns across splits (get_dummies may produce different cols per split)
    train_X, val_X  = train_X.align(val_X,  join="left", axis=1, fill_value=0)
    train_X, test_X = train_X.align(test_X, join="left", axis=1, fill_value=0)
    train_X, oot_X  = train_X.align(oot_X,  join="left", axis=1, fill_value=0)

    print(f"[train] feature matrix shape: {train_X.shape}")
    splits = [("train", train_X, train_y), ("val", val_X, val_y),
              ("test", test_X, test_y), ("oot", oot_X, oot_y)]

    # train + calibrate LR, then tune its decision threshold on val
    lr_raw, lr_params = train_lr(train_X, train_y, val_X, val_y)
    lr_cal = calibrate(lr_raw, val_X, val_y)
    lr_thr = select_threshold(lr_cal, val_X, val_y)
    lr_results = evaluate("lr", lr_cal, splits, threshold=lr_thr)

    # train + calibrate XGB, then tune its decision threshold on val
    xgb_raw, xgb_params = train_xgb(train_X, train_y, val_X, val_y)
    xgb_cal = calibrate(xgb_raw, val_X, val_y)
    xgb_thr = select_threshold(xgb_cal, val_X, val_y)
    xgb_results = evaluate("xgb", xgb_cal, splits, threshold=xgb_thr)

    feature_cols = list(train_X.columns)
    os.makedirs(model_dir, exist_ok=True)

    lr_top10  = get_feature_importances(lr_cal,  feature_cols)
    xgb_top10 = get_feature_importances(xgb_cal, feature_cols)
    print(f"[lr]  top 10 features: {lr_top10}")
    print(f"[xgb] top 10 features: {xgb_top10}")

    candidates = [
        ("lr",  lr_cal,  lr_params,  lr_results,  lr_top10,  lr_thr),
        ("xgb", xgb_cal, xgb_params, xgb_results, xgb_top10, xgb_thr),
    ]

    data_stats = {
        "X_train": train_X.shape[0], "X_val": val_X.shape[0],
        "X_test":  test_X.shape[0],  "X_oot": oot_X.shape[0],
        "y_train": round(float(train_y.mean()), 4),
        "y_val":   round(float(val_y.mean()),   4),
        "y_test":  round(float(test_y.mean()),  4),
        "y_oot":   round(float(oot_y.mean()),   4),
    }

    ds = snapshot_date_str.replace("-", "_")
    for model_name, model, params, results, top10, threshold in candidates:
        artefact = {
            "model":                      model,
            "model_version":              model_name,
            "preprocessing_transformers": {"stdscaler": scaler},
            "data_stats":                 data_stats,
            "results":                    results,
            "hp_params":                  params,
            "feature_cols":               feature_cols,
            "top10_features":             top10,
            "decision_threshold":         threshold,
        }
        artefact_path = os.path.join(model_dir, f"{model_name}_artefact_{ds}.pkl")
        joblib.dump(artefact, artefact_path)
        print(f"[train] {model_name} artefact saved: {artefact_path}")

    return candidates, data_stats, feature_cols, scaler


def select_champion(candidates, data_stats, feature_cols, scaler, snapshot_date_str):
    import joblib

    best_name, best_model, best_params, best_results, best_top10, best_thr = max(
        candidates, key=lambda c: c[3]["oot"]["gini"]
    )
    print(f"\n[champion] selected: {best_name} | OOT gini: {best_results['oot']['gini']} "
          f"| decision_threshold: {best_thr}")

    registry_dir = "model_registry"
    os.makedirs(registry_dir, exist_ok=True)

    version = f"credit_model_{snapshot_date_str.replace('-', '_')}"
    champion = {
        "model":                      best_model,
        "model_version":              version,
        "preprocessing_transformers": {"stdscaler": scaler},
        "data_dates":                 {"model_train_date_str": snapshot_date_str},
        "data_stats": {
            "X_train": data_stats["X_train"],
            "X_val":   data_stats["X_val"],
            "X_test":  data_stats["X_test"],
            "X_oot":   data_stats["X_oot"],
            "y_train": round(data_stats["y_train"], 2),
            "y_val":   round(data_stats["y_val"],   2),
            "y_test":  round(data_stats["y_test"],  2),
            "y_oot":   round(data_stats["y_oot"],   2),
        },
        "results": {
            "gini_train":   best_results["train"]["gini"],
            "gini_val":     best_results["val"]["gini"],
            "gini_test":    best_results["test"]["gini"],
            "gini_oot":     best_results["oot"]["gini"],
            "recall_train": best_results["train"]["recall"],
            "recall_val":   best_results["val"]["recall"],
            "recall_test":  best_results["test"]["recall"],
            "recall_oot":   best_results["oot"]["recall"],
        },
        "hp_params":          best_params,
        "feature_cols":       feature_cols,
        "top10_features":     best_top10,
        "decision_threshold": best_thr,
    }

    out_path = os.path.join(registry_dir, f"{version}.pkl")
    joblib.dump(champion, out_path)
    print(f"[champion] saved: {out_path}")

    # update registry.json
    import json
    registry_path = os.path.join(registry_dir, "registry.json")
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"production": None, "autoML": []}

    if snapshot_date_str == "2024-09-01":
        registry["production"] = version
        print(f"[champion] set as production: {version}")
    else:
        automl_index = len(registry["autoML"]) + 1
        registry["autoML"].append({"name": f"autoML{automl_index}", "version": version})
        print(f"[champion] registered as autoML{automl_index}: {version}")

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[champion] registry updated: {registry_path}")

    return champion


def save_train_baseline(champion, train):
    import joblib

    model       = champion["model"]
    feature_cols = champion["feature_cols"]
    version     = champion["model_version"]

    X_train, _ = get_feature_matrix(train, [c for c in feature_cols if c in train.columns])
    X_train = X_train.reindex(columns=feature_cols, fill_value=0)

    threshold = champion["decision_threshold"]
    scores = model.predict_proba(X_train)[:, 1]
    hard_labels = (scores >= threshold).astype(int)

    # store PSI baseline hard labels + CSI feature distributions in the champion artefact
    registry_path = os.path.join("model_registry", f"{version}.pkl")
    if os.path.exists(registry_path):
        artefact = joblib.load(registry_path)
        artefact["psi_baseline_scores"] = hard_labels

        # CSI: save raw training values for each top-10 feature
        top10 = artefact.get("top10_features", [])
        csi_baseline = {}
        for feat in top10:
            if feat in train.columns:
                csi_baseline[feat] = train[feat].values
        artefact["csi_baseline_distributions"] = csi_baseline

        joblib.dump(artefact, registry_path)
        print(f"[train] PSI baseline + CSI distributions added to artefact: {registry_path}")
        print(f"[train] CSI features tracked: {list(csi_baseline.keys())}")


def train_and_automl(snapshot_date_str, gold_features_dir, gold_label_dir, model_dir, spark):
    df_features = load_feature_store(gold_features_dir, spark)

    # skip retrain when there is no feature data for this snapshot month — past the
    # feature horizon (financials/attributes/clickstream sources end ~2024-12/2025-01,
    # while LMS/labels continue). Nothing new to train on.
    if df_features.filter(col("snapshot_date") == snapshot_date_str).count() == 0:
        print(f"[train] no feature data for {snapshot_date_str} (past feature horizon), skipping retrain.")
        return

    df_labels = load_label_store(gold_label_dir, spark)

    df = build_training_data(df_features, df_labels)
    train, val, test, oot = split_data(df)
    train, val, test, oot, scaler = scale_ema_features(train, val, test, oot, model_dir, snapshot_date_str)
    candidates, data_stats, feature_cols, scaler = train_models(train, val, test, oot, model_dir, scaler, snapshot_date_str)
    champion = select_champion(candidates, data_stats, feature_cols, scaler, snapshot_date_str)
    save_train_baseline(champion, train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("train_and_automl") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    train_and_automl(
        args.snapshotdate,
        "datamart/gold/features/",
        "datamart/gold/label/",
        "model_registry/candidates/",
        spark,
    )

    spark.stop()
