import os
import shutil
import argparse
import random as _random

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType
from datetime import datetime

ALPHA = 0.1
FE_COLS = [f"fe_{i}" for i in range(1, 21)]
EMA_COLS = [f"ema_fe_{i}" for i in range(1, 21)]


def compute_ema(snapshot_date, silver_cs_dir, spark):
    """EMA of each customer's clickstream history, as of snapshot_date.

    The EMA is a per-customer recurrence  EMA_t = a*x_t + (1-a)*EMA_{t-1}  with
    EMA_0 = x_0. It must be folded over the clickstream PANEL (silver_clickstream
    has every customer in every month), because that is the only table that
    retains each customer's monthly history. gold_features cannot supply the
    prior value: its spine (fin_attr) has each customer exactly once, so a
    customer never has a previous-month row there to chain on.

    Computed fresh from the panel each run, so the result does NOT depend on
    prior gold state or on the order months are processed.
    """
    import pandas as pd

    cs_path = os.path.join(silver_cs_dir, "silver_clickstream.parquet")
    pdf = pd.read_parquet(cs_path, columns=["Customer_ID", "snapshot_date"] + FE_COLS)
    # the partition column comes back as an unordered Categorical of date strings;
    # parse to datetime so it compares correctly against the datetime snapshot_date
    pdf["snapshot_date"] = pd.to_datetime(pdf["snapshot_date"].astype(str))
    pdf = pdf[pdf["snapshot_date"] <= snapshot_date]
    print(f"[ema] {snapshot_date} clickstream rows up to snapshot: {len(pdf)}")

    if len(pdf) == 0:
        empty = pd.DataFrame({"Customer_ID": pd.Series(dtype="str"),
                              **{c: pd.Series(dtype="float64") for c in EMA_COLS}})
        return spark.createDataFrame(empty)

    # fold the EMA recurrence over each customer's history, oldest -> newest.
    # ewm(adjust=False) gives exactly  EMA_t = a*x_t + (1-a)*EMA_{t-1}, EMA_0=x_0
    pdf = pdf.sort_values(["Customer_ID", "snapshot_date"])
    for fe, ema_fe in zip(FE_COLS, EMA_COLS):
        pdf[ema_fe] = pdf.groupby("Customer_ID")[fe].transform(
            lambda s: s.ewm(alpha=ALPHA, adjust=False).mean()
        )

    # keep each customer's most recent EMA (their value as of snapshot_date)
    latest = pdf.groupby("Customer_ID").tail(1)[["Customer_ID"] + EMA_COLS]
    return spark.createDataFrame(latest)


def apply_gold_fin_attr_transforms(df, spark):
    """Apply all gold-layer feature engineering to a fin_attr dataframe."""
    total = df.count()

    # Age — impute nulls with median
    age_median = df.select(F.percentile_approx("Age", 0.5)).first()[0]
    null_before = df.filter(col("Age").isNull()).count()
    df = df.withColumn("Age", F.when(col("Age").isNull(), age_median).otherwise(col("Age")).cast(IntegerType()))
    print(f"  Age: {null_before} / {total} nulls imputed with median ({age_median}).")

    # SSN — drop
    df = df.drop("SSN")

    # Occupation — one-hot encode
    occupations = sorted([row["Occupation"] for row in df.select("Occupation").distinct().collect()])
    for occ in occupations:
        df = df.withColumn(f"Occupation_{occ}", F.when(col("Occupation") == occ, 1).otherwise(0).cast(IntegerType()))
    df = df.drop("Occupation")
    print(f"  Occupation: one-hot encoded ({len(occupations)} cols).")

    # Annual_Income — drop
    df = df.drop("Annual_Income")

    # Monthly_Inhand_Salary — keep + bin
    df = df.withColumn("Monthly_Inhand_Salary_binned",
        F.when(col("Monthly_Inhand_Salary") < 1500, "<1500")
         .when(col("Monthly_Inhand_Salary") < 2600, "1500-2600")
         .when(col("Monthly_Inhand_Salary") < 4000, "2600-4000")
         .when(col("Monthly_Inhand_Salary") < 6700, "4000-6700")
         .otherwise(">6700").cast(StringType()))

    # Income_Inconsistent — drop
    df = df.drop("Income_Inconsistent")

    # Num_Bank_Accounts — impute with median
    nba_median = df.select(F.percentile_approx("Num_Bank_Accounts", 0.5)).first()[0]
    df = df.withColumn("Num_Bank_Accounts", F.when(col("Num_Bank_Accounts").isNull(), nba_median).otherwise(col("Num_Bank_Accounts")).cast(IntegerType()))

    # Num_Credit_Card — impute with median
    ncc_median = df.select(F.percentile_approx("Num_Credit_Card", 0.5)).first()[0]
    df = df.withColumn("Num_Credit_Card", F.when(col("Num_Credit_Card").isNull(), ncc_median).otherwise(col("Num_Credit_Card")).cast(IntegerType()))

    # Interest_Rate — impute with median
    ir_median = df.select(F.percentile_approx("Interest_Rate", 0.5)).first()[0]
    df = df.withColumn("Interest_Rate", F.when(col("Interest_Rate").isNull(), ir_median).otherwise(col("Interest_Rate")).cast(IntegerType()))

    # Type_of_Loan — multi-label one-hot encode
    loan_types = sorted(set([
        row["loan_type"][4:] if row["loan_type"].startswith("and ") else row["loan_type"]
        for row in df.select(F.explode(F.split(col("Type_of_Loan"), ", ")).alias("loan_type")).distinct().collect()
        if not (row["loan_type"].startswith("and ") and row["loan_type"][4:] == "Unknown" or row["loan_type"] == "Unknown")
    ]))
    for lt in loan_types:
        df = df.withColumn("Loan_" + lt.replace(" ", "_").replace("-", "_"), F.when(col("Type_of_Loan").contains(lt), 1).otherwise(0).cast(IntegerType()))
    df = df.drop("Type_of_Loan")
    print(f"  Type_of_Loan: one-hot encoded ({len(loan_types)} cols).")

    # Delay_from_due_date — impute with median
    dfdd_median = df.select(F.percentile_approx("Delay_from_due_date", 0.5)).first()[0]
    df = df.withColumn("Delay_from_due_date", F.when(col("Delay_from_due_date").isNull(), dfdd_median).otherwise(col("Delay_from_due_date")).cast(IntegerType()))

    # Num_of_Delayed_Payment — impute with median
    nodp_median = df.select(F.percentile_approx("Num_of_Delayed_Payment", 0.5)).first()[0]
    df = df.withColumn("Num_of_Delayed_Payment", F.when(col("Num_of_Delayed_Payment").isNull(), nodp_median).otherwise(col("Num_of_Delayed_Payment")).cast(IntegerType()))

    # Delay_Inconsistent — bool to int
    df = df.withColumn("Delay_Inconsistent", col("Delay_Inconsistent").cast(IntegerType()))

    # Changed_Credit_Limit — impute with median
    ccl_median = df.select(F.percentile_approx("Changed_Credit_Limit", 0.5)).first()[0]
    df = df.withColumn("Changed_Credit_Limit", F.when(col("Changed_Credit_Limit").isNull(), ccl_median).otherwise(col("Changed_Credit_Limit")).cast(FloatType()))

    # Num_Credit_Inquiries — impute with median
    nci_median = df.select(F.percentile_approx("Num_Credit_Inquiries", 0.5)).first()[0]
    df = df.withColumn("Num_Credit_Inquiries", F.when(col("Num_Credit_Inquiries").isNull(), nci_median).otherwise(col("Num_Credit_Inquiries")).cast(IntegerType()))

    # Credit_Mix — ordinal encode: Bad=0, Unknown=1, Standard=2, Good=3
    df = df.withColumn("Credit_Mix",
        F.when(col("Credit_Mix") == "Bad", 0)
         .when(col("Credit_Mix") == "Unknown", 1)
         .when(col("Credit_Mix") == "Standard", 2)
         .when(col("Credit_Mix") == "Good", 3)
         .otherwise(None).cast(IntegerType()))
    df = df.withColumn("Credit_Mix_Known", F.when(col("Credit_Mix") == 1, 0).otherwise(1).cast(IntegerType()))

    # Outstanding_Debt — keep + bin
    df = df.withColumn("Outstanding_Debt_binned",
        F.when(col("Outstanding_Debt") < 1500, "<1500")
         .when(col("Outstanding_Debt") < 2500, "1500-2500")
         .otherwise(">2500").cast(StringType()))

    # Credit_History_Age — drop (months already parsed in silver)
    df = df.drop("Credit_History_Age")

    # Payment_of_Min_Amount — one-hot encode
    poma_values = sorted([row["Payment_of_Min_Amount"] for row in df.select("Payment_of_Min_Amount").distinct().collect()])
    for v in poma_values:
        df = df.withColumn(f"Payment_of_Min_Amount_{v}", F.when(col("Payment_of_Min_Amount") == v, 1).otherwise(0).cast(IntegerType()))
    df = df.drop("Payment_of_Min_Amount")

    # Total_EMI_per_month — impute with median + bin
    temi_median = df.select(F.percentile_approx("Total_EMI_per_month", 0.5)).first()[0]
    df = df.withColumn("Total_EMI_per_month", F.when(col("Total_EMI_per_month").isNull(), temi_median).otherwise(col("Total_EMI_per_month")).cast(FloatType()))
    df = df.withColumn("Total_EMI_per_month_binned",
        F.when(col("Total_EMI_per_month") < 25, "<25")
         .when(col("Total_EMI_per_month") < 50, "25-50")
         .when(col("Total_EMI_per_month") < 100, "50-100")
         .when(col("Total_EMI_per_month") < 200, "100-200")
         .otherwise(">200").cast(StringType()))

    # Amount_Invested_Monthly — impute with median + bin
    aim_median = df.select(F.percentile_approx("Amount_Invested_Monthly", 0.5)).first()[0]
    df = df.withColumn("Amount_Invested_Monthly", F.when(col("Amount_Invested_Monthly").isNull(), aim_median).otherwise(col("Amount_Invested_Monthly")).cast(FloatType()))
    df = df.withColumn("Amount_Invested_Monthly_binned",
        F.when(col("Amount_Invested_Monthly") < 65, "<65")
         .when(col("Amount_Invested_Monthly") < 110, "65-110")
         .when(col("Amount_Invested_Monthly") < 170, "110-170")
         .when(col("Amount_Invested_Monthly") < 320, "170-320")
         .otherwise(">320").cast(StringType()))

    # Spent_Level — replace 'Unknown' via proportional sampling, ordinal encode (Low=0, High=1)
    known_counts = {row["Spent_Level"]: row["count"] for row in df.filter(col("Spent_Level") != "Unknown").groupBy("Spent_Level").count().collect()}
    total_known = sum(known_counts.values())
    proportions_sl = {k: v / total_known for k, v in known_counts.items()}
    values_sl = list(proportions_sl.keys())
    weights_sl = [proportions_sl[v] for v in values_sl]
    bc_sl = spark.sparkContext.broadcast((values_sl, weights_sl))

    @F.udf(StringType())
    def sample_spent_level(_):
        v, w = bc_sl.value
        return _random.choices(v, weights=w, k=1)[0]

    df = df.withColumn("Spent_Level", F.when(col("Spent_Level") == "Unknown", sample_spent_level(F.lit(0))).otherwise(col("Spent_Level")))
    df = df.withColumn("Spent_Level",
        F.when(col("Spent_Level") == "Low", 0).when(col("Spent_Level") == "High", 1).otherwise(None).cast(IntegerType()))

    # Value_Payments — replace 'Unknown' via proportional sampling, ordinal encode (Small=0, Medium=1, Large=2)
    known_counts_vp = {row["Value_Payments"]: row["count"] for row in df.filter(col("Value_Payments") != "Unknown").groupBy("Value_Payments").count().collect()}
    total_known_vp = sum(known_counts_vp.values())
    proportions_vp = {k: v / total_known_vp for k, v in known_counts_vp.items()}
    values_vp = list(proportions_vp.keys())
    weights_vp = [proportions_vp[v] for v in values_vp]
    bc_vp = spark.sparkContext.broadcast((values_vp, weights_vp))

    @F.udf(StringType())
    def sample_value_payments(_):
        v, w = bc_vp.value
        return _random.choices(v, weights=w, k=1)[0]

    df = df.withColumn("Value_Payments", F.when(col("Value_Payments") == "Unknown", sample_value_payments(F.lit(0))).otherwise(col("Value_Payments")))
    df = df.withColumn("Value_Payments",
        F.when(col("Value_Payments") == "Small", 0).when(col("Value_Payments") == "Medium", 1).when(col("Value_Payments") == "Large", 2).otherwise(None).cast(IntegerType()))

    # Monthly_Balance — drop (high correlation with Monthly_Inhand_Salary)
    df = df.drop("Monthly_Balance")

    return df


def process_gold_features(snapshot_date_str, silver_fin_attr_dir, silver_cs_dir, gold_features_dir, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    out_path = os.path.join(gold_features_dir, "gold_features.parquet")

    # 1. Compute EMA using previous month's rows from gold_features
    df_ema = compute_ema(snapshot_date, silver_cs_dir, spark)

    # 2. Load this month's fin_attr partition from silver
    fin_attr_path = os.path.join(silver_fin_attr_dir, "silver_fin_attr.parquet")
    df_fin_attr = spark.read.parquet(fin_attr_path).filter(col("snapshot_date") == snapshot_date)
    print(f"[gold_features] {snapshot_date_str} fin_attr rows: {df_fin_attr.count()}")

    # 3. Apply gold-layer feature engineering
    df_features = apply_gold_fin_attr_transforms(df_fin_attr, spark)

    # 4. Left join EMA on Customer_ID + flag customers with no clickstream history
    df_features = df_features.join(df_ema, on="Customer_ID", how="left")
    df_features = df_features.withColumn(
        "has_clickstream",
        F.when(col("ema_fe_1").isNotNull(), 1).otherwise(0).cast(IntegerType())
    )

    print(f"[gold_features] {snapshot_date_str} | rows: {df_features.count()} | cols: {len(df_features.columns)}")

    # 5. Upsert into flat single table: drop this month if re-run, then append
    os.makedirs(gold_features_dir, exist_ok=True)
    marker = os.path.join(gold_features_dir, f".done_{snapshot_date_str}")

    if os.path.exists(marker):
        print(f"[gold_features] {snapshot_date_str} already processed, skipping.")
        return

    if os.path.exists(out_path):
        try:
            df_existing = spark.read.parquet(out_path).filter(col("snapshot_date") != snapshot_date)
            df_out = df_existing.unionByName(df_features)
        except Exception:
            # parquet dir exists but is empty/corrupt (e.g. from a prior failed run)
            df_out = df_features
    else:
        df_out = df_features
    # write to a temp path then swap in. Spark cannot safely read and overwrite
    # the same path in one action — overwrite deletes the destination before the
    # lazy read (both the existing-table union and the EMA lookup) executes,
    # emptying/corrupting the table.
    tmp_path = out_path + "_tmp"
    # clear any stale temp dir from a prior failed attempt with Python — Spark's
    # own overwrite can't clear it through the macOS FUSE mount.
    shutil.rmtree(tmp_path, ignore_errors=True)
    df_out.write.mode("overwrite").parquet(tmp_path)
    shutil.rmtree(out_path, ignore_errors=True)
    os.rename(tmp_path, out_path)
    open(marker, "w").close()
    print(f"[gold_features] saved: {out_path}")

    return df_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_features") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_gold_features(
        args.snapshotdate,
        "datamart/silver/fin_attr/",
        "datamart/silver/clickstream/",
        "datamart/gold/features/",
        spark,
    )

    spark.stop()
