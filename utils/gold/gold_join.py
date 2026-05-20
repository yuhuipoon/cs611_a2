import os
import glob
import random as _random

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.window import Window


def process_gold_join(gold_features_dir, gold_label_dir, gold_joined_dir, spark):
    # load features store (Customer_ID, fin_attr_snapshot_date, [fin_attr cols], cs_snapshot_date, fe_1..fe_20)
    df_features = spark.read.parquet(gold_features_dir + "gold_features.parquet")
    print('loaded features from:', gold_features_dir + "gold_features.parquet", '| row count:', df_features.count())

    # load all label store partitions
    partition_paths = sorted(glob.glob(os.path.join(gold_label_dir, "*.parquet")))
    df_label = spark.read.parquet(*partition_paths)
    df_label = df_label.withColumnRenamed("snapshot_date", "label_snapshot_date")
    print(f'loaded labels from: {gold_label_dir} | partitions: {len(partition_paths)} | row count:', df_label.count())

    # 1) join on Customer_ID — df_features is left to preserve customers without labels
    df = df_features.join(df_label, on="Customer_ID", how="left")
    print(f"  After join: {df.count()} rows, {len(df.columns)} cols.")

    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    alpha = 0.1

    # 2) keep only cs rows where cs_snapshot_date <= label_snapshot_date (nulls kept for no-cs customers)
    df = df.filter(col("cs_snapshot_date").isNull() | (col("cs_snapshot_date") <= col("label_snapshot_date")))
    print(f"  After cs date filter: {df.count()} rows.")

    # split: customers with and without valid clickstream
    df_has_cs = df.filter(col("cs_snapshot_date").isNotNull())
    df_no_cs  = df.filter(col("cs_snapshot_date").isNull()).drop("cs_snapshot_date")

    # EWM: rank each cs row within (Customer_ID, fin_attr_snapshot_date, label_snapshot_date) ascending by date
    group_keys = ["Customer_ID", "fin_attr_snapshot_date", "label_snapshot_date"]
    other_cols = [c for c in df_has_cs.columns if c not in group_keys + fe_cols + ["cs_snapshot_date"]]

    w = Window.partitionBy(*group_keys).orderBy("cs_snapshot_date")
    df_has_cs = df_has_cs.withColumn("_rank", F.row_number().over(w) - 1)
    df_has_cs = df_has_cs.withColumn("_weight", F.exp(F.lit(alpha) * col("_rank")))
    for c in fe_cols:
        df_has_cs = df_has_cs.withColumn(f"_w_{c}", col("_weight") * col(c).cast(IntegerType()))

    agg_exprs = (
        [F.sum("_weight").alias("_sum_weight")] +
        [F.first(c).alias(c) for c in other_cols] +
        [F.sum(f"_w_{c}").alias(f"_wsum_{c}") for c in fe_cols]
    )
    df_cs_agg = df_has_cs.groupBy(*group_keys).agg(*agg_exprs)
    for c in fe_cols:
        df_cs_agg = df_cs_agg.withColumn(c, F.round(col(f"_wsum_{c}") / col("_sum_weight")).cast(IntegerType())).drop(f"_wsum_{c}")
    df_cs_agg = df_cs_agg.drop("_sum_weight")
    print(f"  EWM-aggregated clickstream rows: {df_cs_agg.count()}")

    # 3) impute fe_1..fe_20 for no-cs customers via column-wise distributional sampling
    def make_sampler(bc):
        @F.udf(FloatType())
        def _sample(_):
            return float(_random.choice(bc.value))
        return _sample

    n_imputed = df_no_cs.count()
    for c in fe_cols:
        values = [row[c] for row in df_cs_agg.select(c).collect() if row[c] is not None]
        sampler = make_sampler(spark.sparkContext.broadcast(values))
        df_no_cs = df_no_cs.withColumn(c, sampler(F.lit(0)).cast(IntegerType()))
    print(f"  Imputed fe_1..fe_20 for {n_imputed} no-cs rows via distributional sampling.")

    df = df_cs_agg.unionByName(df_no_cs)
    print(f"  After EWM agg + imputation: {df.count()} rows.")

    # 4) drop leakage: fin_attr_snapshot_date >= label_snapshot_date
    flagged_count = df.filter(col("fin_attr_snapshot_date") >= col("label_snapshot_date")).count()
    if flagged_count > 0:
        print(f"  WARNING: {flagged_count} rows dropped where fin_attr_snapshot_date >= label_snapshot_date.")
    else:
        print(f"  OK: no rows with fin_attr_snapshot_date >= label_snapshot_date.")
    df = df.filter(col("fin_attr_snapshot_date") < col("label_snapshot_date"))
    print(f"  After leakage filter: {df.count()} rows, {len(df.columns)} cols.")

    # enforce column order
    occ_cols  = sorted([c for c in df.columns if c.startswith("Occupation_")])
    loan_cols = sorted([c for c in df.columns if c.startswith("Loan_")])
    poma_cols = sorted([c for c in df.columns if c.startswith("Payment_of_Min_Amount_")])

    ordered_cols = (
        ["Customer_ID", "label", "Age"] +
        occ_cols +
        ["Monthly_Inhand_Salary", "Monthly_Inhand_Salary_binned",
         "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
         "Num_of_Loan"] +
        loan_cols +
        ["Delay_from_due_date", "Num_of_Delayed_Payment", "Delay_Inconsistent",
         "Changed_Credit_Limit", "Num_Credit_Inquiries",
         "Credit_Mix", "Credit_Mix_Known",
         "Outstanding_Debt", "Outstanding_Debt_binned",
         "Credit_Utilization_Ratio", "Credit_History_Age_Months"] +
        poma_cols +
        ["Total_EMI_per_month", "Total_EMI_per_month_binned",
         "Amount_Invested_Monthly", "Amount_Invested_Monthly_binned",
         "Spent_Level", "Value_Payments"] +
        fe_cols +
        ["has_clickstream"]
    )
    df = df.select(ordered_cols)

    output_filepath = gold_joined_dir + "features_label_join.parquet"
    df.write.mode("overwrite").parquet(output_filepath)
    print('saved to:', output_filepath)

    return df
