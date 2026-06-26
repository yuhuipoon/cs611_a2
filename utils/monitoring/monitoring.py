import os
import argparse

import pandas as pd
import pyspark

META_COLS = ["Customer_ID", "loan_id", "snapshot_date", "label_date", "label", "label_def"]


def load_inference_batch(inference_dir, snapshot_date_str, model_label="production"):
    batch_dir = os.path.join(inference_dir, f"batch_{snapshot_date_str}")
    path = os.path.join(batch_dir, f"{model_label}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    print(f"[monitor] loaded {model_label} inference: {snapshot_date_str} | rows: {len(df)}")
    return df


def list_automl_files(inference_dir, snapshot_date_str):
    batch_dir = os.path.join(inference_dir, f"batch_{snapshot_date_str}")
    if not os.path.exists(batch_dir):
        return []
    return [
        f.replace(".parquet", "")
        for f in os.listdir(batch_dir)
        if f.startswith("autoML") and f.endswith(".parquet")
    ]


def load_label_store(gold_label_dir):
    import pandas as pd
    path = os.path.join(gold_label_dir, "gold_label_store.parquet")
    df = pd.read_parquet(path, columns=["Customer_ID", "label"])
    print(f"[monitor] label store loaded | rows: {len(df)}")
    return df


def compute_metrics(y_true, y_prob, y_pred=None, threshold=0.5):
    from sklearn.metrics import roc_auc_score, recall_score
    if len(y_true) == 0 or y_true.nunique() < 2:
        return None
    auc = roc_auc_score(y_true, y_prob)
    # prefer the stored prediction (made at the model's tuned threshold); fall back
    # to thresholding the score for older batches that have no prediction column.
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    return {
        "gini":   round(2 * auc - 1, 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "n":      len(y_true),
    }


def compute_csi(expected_vals, actual_vals, n_bins=10):
    import numpy as np
    import pandas as pd

    exp = pd.Series(expected_vals).dropna()
    act = pd.Series(actual_vals).dropna()
    if len(exp) == 0 or len(act) == 0:
        return float("nan")

    # Discrete feature (non-numeric, or few distinct values — e.g. one-hot dummies,
    # ordinal codes like Credit_Mix) -> one bin per category. Quantile bins collapse
    # on these (a 2-value column has degenerate percentiles -> CSI ~0). Continuous
    # features still use quantile bins from the training distribution.
    if (not np.issubdtype(exp.dtype, np.number)) or (exp.nunique() <= n_bins):
        cats = sorted(set(exp.unique()) | set(act.unique()), key=str)
        expected_pct = np.array([float((exp == c).mean()) for c in cats])
        actual_pct   = np.array([float((act == c).mean()) for c in cats])
    else:
        bins = np.unique(np.nanpercentile(exp, np.linspace(0, 100, n_bins + 1)))
        bins[0]  = -np.inf
        bins[-1] =  np.inf
        expected_pct = np.histogram(exp, bins=bins)[0] / len(exp)
        actual_pct   = np.histogram(act, bins=bins)[0] / len(act)

    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)

    csi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(csi), 4)


def compute_psi(expected_labels, actual_labels):
    import numpy as np
    # Hard-label PSI: two bins — 0 (no default) and 1 (default predicted).
    expected_pct = np.array([
        (expected_labels == 0).mean(),
        (expected_labels == 1).mean(),
    ])
    actual_pct = np.array([
        (actual_labels == 0).mean(),
        (actual_labels == 1).mean(),
    ])

    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def run_monitor(snapshot_date_str, inference_dir, gold_label_dir, spark):
    # find all inference batches whose labels may have matured by now
    if not os.path.exists(inference_dir):
        print(f"[monitor] inference dir not found, skipping.")
        return

    batch_dates = sorted([
        d.replace("batch_", "")
        for d in os.listdir(inference_dir)
        if d.startswith("batch_")
    ])

    if not batch_dates:
        print(f"[monitor] no inference batches found, skipping.")
        return

    df_labels = load_label_store(gold_label_dir)

    # PSI and CSI baselines BOTH come from the PRODUCTION model's artefact, so they
    # stay fixed to the production model. (The shared psi_csi_train_baseline.parquet
    # is overwritten by every autoML retrain, which would make a given month's PSI
    # drift across monitor runs — the baseline, not the batch, would be moving.)
    import json
    import joblib
    df_baseline = None
    csi_baseline = {}
    registry_path = "model_registry/registry.json"
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry_meta = json.load(f)
        prod_version = registry_meta.get("production")
        if prod_version:
            champ_path = os.path.join("model_registry", f"{prod_version}.pkl")
            if os.path.exists(champ_path):
                champ = joblib.load(champ_path)
                psi_scores = champ.get("psi_baseline_scores")
                if psi_scores is not None:
                    df_baseline = pd.DataFrame({"prediction": psi_scores})
                    print(f"[monitor] PSI baseline loaded from production artefact "
                          f"({prod_version}) | rows: {len(df_baseline)}")
                csi_baseline = champ.get("csi_baseline_distributions", {})
                print(f"[monitor] CSI baseline loaded | features: {list(csi_baseline.keys())}")
    if df_baseline is None:
        print(f"[monitor] PSI baseline not found, PSI will be skipped.")

    production_records = []
    automl_records = []
    psi_records = []
    csi_records = []

    for batch_date in batch_dates:
        batch_dir = os.path.join(inference_dir, f"batch_{batch_date}")

        # --- production ---
        df_inf = load_inference_batch(inference_dir, batch_date, "production")
        if df_inf is not None:
            # PSI — no label lag needed, compute immediately
            if df_baseline is not None:
                psi = compute_psi(df_baseline["prediction"].values, df_inf["prediction"].values)
                print(f"[monitor] PSI production {batch_date} | psi: {psi}")
                psi_records.append({
                    "inference_date": batch_date,
                    "monitor_date":   snapshot_date_str,
                    "model_label":    "production",
                    "psi":            psi,
                })

            # CSI — no label lag needed, compute immediately
            if csi_baseline:
                csi_row = {"inference_date": batch_date, "monitor_date": snapshot_date_str}
                for feat, baseline_vals in csi_baseline.items():
                    if feat in df_inf.columns:
                        csi_val = compute_csi(baseline_vals, df_inf[feat].dropna().values)
                        csi_row[feat] = csi_val
                        print(f"[monitor] CSI {feat} {batch_date} | csi: {csi_val}")
                csi_records.append(csi_row)

            # performance — requires matured labels
            df_joined = df_inf.merge(df_labels, on="Customer_ID", how="inner")
            if len(df_joined) > 0:
                metrics = compute_metrics(df_joined["label"], df_joined["score"],
                                      y_pred=df_joined["prediction"] if "prediction" in df_joined.columns else None)
                if metrics:
                    print(f"[monitor] production {batch_date} | gini: {metrics['gini']} | recall: {metrics['recall']}")
                    production_records.append({
                        "inference_date": batch_date,
                        "monitor_date":   snapshot_date_str,
                        "model_version":  df_joined["model_version"].iloc[0],
                        "model_label":    "production",
                        "n":              metrics["n"],
                        "gini":           metrics["gini"],
                        "recall":         metrics["recall"],
                    })
            else:
                print(f"[monitor] production {batch_date} — labels not yet mature, skipping perf metrics.")

            # save per-batch monitor outputs into the batch directory
            batch_psi = [r for r in psi_records if r["inference_date"] == batch_date]
            if batch_psi:
                psi_path = os.path.join(batch_dir, "monitor_psi.parquet")
                pd.DataFrame(batch_psi).to_parquet(psi_path, index=False)
                print(f"[monitor] PSI saved: {psi_path}")

            batch_csi = [r for r in csi_records if r["inference_date"] == batch_date]
            if batch_csi:
                csi_path = os.path.join(batch_dir, "monitor_csi.parquet")
                pd.DataFrame(batch_csi).to_parquet(csi_path, index=False)
                print(f"[monitor] CSI saved: {csi_path}")

            batch_perf = [r for r in production_records if r["inference_date"] == batch_date]

        # --- autoML (one per batch at most) ---
        for automl_label in list_automl_files(inference_dir, batch_date):
            df_aml = load_inference_batch(inference_dir, batch_date, automl_label)
            if df_aml is None:
                continue
            df_joined = df_aml.merge(df_labels, on="Customer_ID", how="inner")
            if len(df_joined) == 0:
                print(f"[monitor] {automl_label} {batch_date} — labels not yet mature, skipping.")
                continue
            metrics = compute_metrics(df_joined["label"], df_joined["score"],
                                      y_pred=df_joined["prediction"] if "prediction" in df_joined.columns else None)
            if metrics:
                print(f"[monitor] {automl_label} {batch_date} | gini: {metrics['gini']} | recall: {metrics['recall']}")
                automl_records.append({
                    "inference_date": batch_date,
                    "monitor_date":   snapshot_date_str,
                    "model_version":  df_joined["model_version"].iloc[0],
                    "model_label":    automl_label,
                    "n":              metrics["n"],
                    "gini":           metrics["gini"],
                    "recall":         metrics["recall"],
                })

        # save combined performance (production + autoML) for this batch
        batch_automl = [r for r in automl_records if r["inference_date"] == batch_date]
        batch_perf = [r for r in production_records if r["inference_date"] == batch_date]
        all_perf = batch_perf + batch_automl
        if all_perf:
            perf_path = os.path.join(batch_dir, "monitor_performance.parquet")
            pd.DataFrame(all_perf).to_parquet(perf_path, index=False)
            print(f"[monitor] performance saved: {perf_path}")

    if not production_records and not automl_records and not psi_records and not csi_records:
        print(f"[monitor] nothing to report on {snapshot_date_str}.")
        return

    # plots aggregate across all batches — write into the snapshot's own batch dir
    snapshot_batch_dir = os.path.join(inference_dir, f"batch_{snapshot_date_str}")
    os.makedirs(snapshot_batch_dir, exist_ok=True)

    plot_performance(pd.DataFrame(production_records), pd.DataFrame(automl_records),
                     snapshot_date_str, snapshot_batch_dir)
    if psi_records:
        df_psi = pd.DataFrame(psi_records)
        plot_psi(df_psi, snapshot_date_str, snapshot_batch_dir, inference_dir, df_baseline)
    if csi_records:
        df_csi = pd.DataFrame(csi_records)
        plot_csi(df_csi, snapshot_date_str, snapshot_batch_dir, list(csi_baseline.keys()))


def plot_performance(df_prod, df_automl, snapshot_date_str, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # thresholds
    GINI_GREEN  = 0.65
    GINI_YELLOW = 0.60
    RECALL_GREEN  = 0.65
    RECALL_YELLOW = 0.60

    from dateutil.relativedelta import relativedelta
    from datetime import datetime

    # x-axis is the MEASUREMENT timeline: a performance point only appears the month
    # its labels mature (inference + 6 months). Plotting each point at its matured
    # month makes the 6-month warm-up (no measurable performance) visibly empty.
    start = datetime(2024, 10, 1)
    end   = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    all_months = []
    cur = start
    while cur <= end:
        all_months.append(cur.strftime("%Y-%m-%d"))
        cur += relativedelta(months=1)

    x_all = list(range(len(all_months)))
    date_to_x = {d: i for i, d in enumerate(all_months)}

    def matured_x(inference_date_str):
        # x-position = the month this batch's labels matured (inference + 6 months)
        m = datetime.strptime(inference_date_str, "%Y-%m-%d") + relativedelta(months=6)
        return date_to_x.get(m.strftime("%Y-%m-%d"))

    def perf_label(m_str):
        m = datetime.strptime(m_str, "%Y-%m-%d")
        inf = m - relativedelta(months=6)
        # top: month performance is measured; bottom: the inference batch it scores
        # (6 months earlier). Only label a batch if an inference actually ran then.
        if inf >= start:
            return f"{m.strftime('%b %Y')}\n↑ scores\n{inf.strftime('%b %Y')}"
        return f"{m.strftime('%b %Y')}\n\n"

    x_labels = [perf_label(d) for d in all_months]

    # first month any label can mature = first inference (Oct 2024) + 6 months
    first_maturity_x = date_to_x.get(
        (start + relativedelta(months=6)).strftime("%Y-%m-%d"), len(all_months))

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle(f"Model Performance (monitor run: {snapshot_date_str})", fontsize=12)

    def draw_bands(ax, green_thresh, yellow_thresh):
        ax.axhspan(green_thresh,  1.0,          alpha=0.15, color="green")
        ax.axhspan(yellow_thresh, green_thresh,  alpha=0.15, color="yellow")
        ax.axhspan(0.0,           yellow_thresh, alpha=0.15, color="red")
        ax.set_ylim(0.2, 1.0)
        ax.set_xlim(-0.5, len(all_months) - 0.5)
        # grey out the warm-up: months before any labels could mature (no points yet)
        if first_maturity_x > 0:
            ax.axvspan(-0.5, first_maturity_x - 0.5, color="lightgray", alpha=0.45, zorder=4)
            ax.text((first_maturity_x - 0.5) / 2.0 - 0.25, 0.5,
                    "no matured ground-truth labels yet\n(6-month label lag)",
                    ha="center", va="center", fontsize=8, color="dimgray",
                    style="italic", zorder=5)

    def plot_automl_points(ax, metric):
        for _, row in df_automl.iterrows():
            xi = matured_x(row["inference_date"])   # plot at maturity month
            if xi is None:
                continue
            ax.scatter(xi, row[metric], marker="*", s=100, color="steelblue", zorder=6)
            ax.annotate(f"{row[metric]:.3f} ({row['model_label']})",
                        (xi, row[metric]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5, color="steelblue")

    def plot_prod_line(ax, metric):
        if len(df_prod) == 0:
            return
        pts = [(matured_x(d), df_prod.loc[df_prod["inference_date"] == d, metric].values[0])
               for d in df_prod["inference_date"]]
        pts = sorted((x, y) for x, y in pts if x is not None)
        if not pts:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color="black", linewidth=2, zorder=6)
        for xi, yi in zip(xs, ys):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8)

    # P0 — Gini
    ax0 = axes[0]
    draw_bands(ax0, GINI_GREEN, GINI_YELLOW)
    plot_prod_line(ax0, "gini")
    if len(df_automl) > 0:
        plot_automl_points(ax0, "gini")
    ax0.axhline(GINI_GREEN,  color="green",  linestyle="--", linewidth=0.8)
    ax0.axhline(GINI_YELLOW, color="orange", linestyle="--", linewidth=0.8)
    ax0.set_title("P0 — Gini Coefficient", fontsize=10)
    ax0.set_ylabel("Gini", fontsize=9)
    ax0.tick_params(axis="y", labelsize=8)
    ax0.set_xticks(x_all)
    ax0.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=7)

    # P1 — Recall
    ax1 = axes[1]
    draw_bands(ax1, RECALL_GREEN, RECALL_YELLOW)
    plot_prod_line(ax1, "recall")
    if len(df_automl) > 0:
        plot_automl_points(ax1, "recall")
    ax1.axhline(RECALL_GREEN,  color="green",  linestyle="--", linewidth=0.8)
    ax1.axhline(RECALL_YELLOW, color="orange", linestyle="--", linewidth=0.8)
    ax1.set_title("P1 — Recall", fontsize=10)
    ax1.set_ylabel("Recall", fontsize=9)
    ax1.tick_params(axis="y", labelsize=8)
    ax1.set_xticks(x_all)
    ax1.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=7)
    ax1.set_xlabel(
        "tick = month performance is measured  ↑  inference batch it evaluates "
        "(6 months earlier)",
        fontsize=8,
    )

    # shared legend
    legend = [
        mpatches.Patch(color="green",  alpha=0.4, label="Green"),
        mpatches.Patch(color="yellow", alpha=0.4, label="Yellow"),
        mpatches.Patch(color="red",    alpha=0.4, label="Red"),
        plt.Line2D([0], [0], color="black", marker="o", linewidth=2, label="Production"),
        plt.Line2D([0], [0], color="steelblue", marker="*", markersize=10, linewidth=0, label="AutoML"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    png_path = os.path.join(out_dir, f"performance_{snapshot_date_str}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitor] plot saved: {png_path}")


def plot_psi(df_psi, snapshot_date_str, out_dir, inference_dir, df_baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from dateutil.relativedelta import relativedelta
    from datetime import datetime

    PSI_GREEN  = 0.10
    PSI_YELLOW = 0.25

    start = datetime(2024, 10, 1)
    # cap the x-axis at the last inference month, not the monitor date — inferences
    # stop at the feature horizon, so PSI only exists while batches were scored.
    last_inf = max(df_psi["inference_date"]) if len(df_psi) else snapshot_date_str
    end   = datetime.strptime(last_inf, "%Y-%m-%d")
    all_months = []
    cur = start
    while cur <= end:
        all_months.append(cur.strftime("%Y-%m-%d"))
        cur += relativedelta(months=1)

    x_all = list(range(len(all_months)))
    date_to_x = {d: i for i, d in enumerate(all_months)}
    x_labels = [datetime.strptime(d, "%Y-%m-%d").strftime("%b %Y") for d in all_months]

    # rows: 1 PSI line + 1 distribution plot for today's batch only
    n_rows = 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 7))
    fig.suptitle(f"Stability — PSI (monitor run: {snapshot_date_str})", fontsize=12)

    ax = axes[0] if n_rows > 1 else axes

    # --- PSI line chart ---
    xs = [date_to_x[d] for d in df_psi["inference_date"] if d in date_to_x]
    ys = [df_psi.loc[df_psi["inference_date"] == d, "psi"].values[0]
          for d in df_psi["inference_date"] if d in date_to_x]

    # PSI can exceed 1.0 under major drift; grow the y-axis to fit the points,
    # otherwise they get clipped above a fixed 0.5 cap and the line disappears.
    top = max(0.5, (max(ys) * 1.15) if ys else 0.5)
    ax.axhspan(0,          PSI_GREEN,  alpha=0.15, color="green")
    ax.axhspan(PSI_GREEN,  PSI_YELLOW, alpha=0.15, color="yellow")
    ax.axhspan(PSI_YELLOW, top,        alpha=0.15, color="red")
    ax.set_ylim(0, top)
    ax.set_xlim(-0.5, len(all_months) - 0.5)
    ax.axhline(PSI_GREEN,  color="green",  linestyle="--", linewidth=0.8)
    ax.axhline(PSI_YELLOW, color="orange", linestyle="--", linewidth=0.8)

    ax.plot(xs, ys, marker="o", color="black", linewidth=2, zorder=5)
    for xi, yi in zip(xs, ys):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)

    ax.set_ylabel("PSI", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xticks(x_all)
    ax.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=7.5)

    legend = [
        mpatches.Patch(color="green",  alpha=0.4, label="Green (<0.10)"),
        mpatches.Patch(color="yellow", alpha=0.4, label="Yellow (0.10–0.25)"),
        mpatches.Patch(color="red",    alpha=0.4, label="Red (>0.25)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=7.5)

    # --- hard-label distribution for today's batch ---
    import numpy as np
    baseline_labels = df_baseline["prediction"].values if df_baseline is not None else None

    ax_dist = axes[1]
    batch_path = os.path.join(inference_dir, f"batch_{snapshot_date_str}", "production.parquet")
    if os.path.exists(batch_path) and baseline_labels is not None:
        batch_labels = pd.read_parquet(batch_path)["prediction"].values
        psi_val = df_psi.loc[df_psi["inference_date"] == snapshot_date_str, "psi"].values
        psi_val = psi_val[0] if len(psi_val) > 0 else float("nan")

        x = np.array([0, 1])
        width = 0.35
        baseline_pct = np.array([(baseline_labels == 0).mean(), (baseline_labels == 1).mean()])
        batch_pct    = np.array([(batch_labels    == 0).mean(), (batch_labels    == 1).mean()])

        ax_dist.bar(x - width / 2, baseline_pct, width, alpha=0.6, color="steelblue", label="Train (expected)")
        ax_dist.bar(x + width / 2, batch_pct,    width, alpha=0.6, color="tomato",    label="Inference (actual)")
        ax_dist.set_xticks(x)
        ax_dist.set_xticklabels(["No Default (0)", "Default (1)"], fontsize=8)
        ax_dist.set_title(f"Prediction Distribution — {snapshot_date_str}  |  PSI: {psi_val:.3f}", fontsize=9)
        ax_dist.set_xlabel("Predicted Label", fontsize=8)
        ax_dist.set_ylabel("Proportion", fontsize=8)
        ax_dist.set_ylim(0, 1)
        ax_dist.tick_params(axis="y", labelsize=7.5)
        ax_dist.legend(fontsize=7.5)
    else:
        ax_dist.set_visible(False)

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"psi_{snapshot_date_str}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitor] PSI plot saved: {png_path}")


def plot_csi(df_csi, snapshot_date_str, out_dir, feature_cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from dateutil.relativedelta import relativedelta
    from datetime import datetime

    CSI_GREEN  = 0.10
    CSI_YELLOW = 0.25

    start = datetime(2024, 10, 1)
    # cap the x-axis at the last inference month, not the monitor date — inferences
    # stop at the feature horizon, so CSI only exists while batches were scored.
    last_inf = max(df_csi["inference_date"]) if len(df_csi) else snapshot_date_str
    end   = datetime.strptime(last_inf, "%Y-%m-%d")
    all_months = []
    cur = start
    while cur <= end:
        all_months.append(cur.strftime("%Y-%m-%d"))
        cur += relativedelta(months=1)

    x_all = list(range(len(all_months)))
    date_to_x = {d: i for i, d in enumerate(all_months)}
    x_labels = [datetime.strptime(d, "%Y-%m-%d").strftime("%b %Y") for d in all_months]

    n_features = len(feature_cols)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    fig.suptitle(f"Stability — CSI by Feature (monitor run: {snapshot_date_str})", fontsize=12)
    axes_flat = axes.flatten()

    for i, feat in enumerate(feature_cols):
        ax = axes_flat[i]
        ax.set_title(feat, fontsize=9)
        if feat not in df_csi.columns:
            ax.set_visible(False)
            continue

        xs = [date_to_x[d] for d in df_csi["inference_date"] if d in date_to_x]
        ys = [df_csi.loc[df_csi["inference_date"] == d, feat].values[0]
              for d in df_csi["inference_date"] if d in date_to_x]

        # all-NaN feature (e.g. EMA: scored cohorts have no clickstream) — say so
        # instead of drawing an empty band chart with no line
        if len(ys) == 0 or all(pd.isna(v) for v in ys):
            ax.text(0.5, 0.5, "no inference data\n(feature unavailable\nfor production cohorts)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        ax.axhspan(0,         CSI_GREEN,  alpha=0.15, color="green")
        ax.axhspan(CSI_GREEN, CSI_YELLOW, alpha=0.15, color="yellow")
        ax.axhspan(CSI_YELLOW, 0.5,       alpha=0.15, color="red")
        ax.set_ylim(0, 0.5)
        ax.set_xlim(-0.5, len(all_months) - 0.5)
        ax.axhline(CSI_GREEN,  color="green",  linestyle="--", linewidth=0.8)
        ax.axhline(CSI_YELLOW, color="orange", linestyle="--", linewidth=0.8)

        ax.plot(xs, ys, marker="o", color="black", linewidth=1.5, zorder=5)
        for xi, yi in zip(xs, ys):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7)

        ax.set_ylabel("CSI", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xticks(x_all)
        ax.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=6.5)

    # hide any unused subplots
    for j in range(n_features, len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend = [
        mpatches.Patch(color="green",  alpha=0.4, label="Green (<0.10)"),
        mpatches.Patch(color="yellow", alpha=0.4, label="Yellow (0.10–0.25)"),
        mpatches.Patch(color="red",    alpha=0.4, label="Red (>0.25)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    png_path = os.path.join(out_dir, f"csi_{snapshot_date_str}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitor] CSI plot saved: {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("model_monitor") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    run_monitor(
        args.snapshotdate,
        "datamart/gold/inference/",
        "datamart/gold/label/",
        spark,
    )

    spark.stop()
