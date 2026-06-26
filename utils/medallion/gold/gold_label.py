import os
import shutil
import argparse

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType


def process_gold_label(snapshot_date_str, silver_lms_dir, gold_label_dir, spark, dpd=30, mob=6):
    partition_name = f"silver_lms_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_lms_dir, partition_name)
    df = spark.read.parquet(filepath)
    print(f"[gold_label] loaded: {filepath} | rows: {df.count()}")

    # loans that have reached the target MOB this month
    df = df.filter(col("mob") == mob)

    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    print(f"[gold_label] {snapshot_date_str} | mob={mob} loans: {df.count()} | defaults: {df.filter(col('label') == 1).count()}")

    # upsert into flat single table
    os.makedirs(gold_label_dir, exist_ok=True)
    out_path = os.path.join(gold_label_dir, "gold_label_store.parquet")
    marker = os.path.join(gold_label_dir, f".done_{snapshot_date_str}")

    if os.path.exists(marker):
        print(f"[gold_label] {snapshot_date_str} already processed, skipping.")
        return

    if os.path.exists(out_path):
        try:
            df_existing = spark.read.parquet(out_path).filter(col("snapshot_date") != snapshot_date_str)
            df_out = df_existing.unionByName(df)
        except Exception:
            # parquet dir exists but is empty/corrupt (e.g. from a prior failed run)
            df_out = df
    else:
        df_out = df
    # write to a temp path then swap in. Spark cannot safely read and overwrite
    # the same path in one action — overwrite deletes the destination before the
    # lazy read executes, emptying/corrupting the table.
    tmp_path = out_path + "_tmp"
    # clear any stale temp dir from a prior failed attempt with Python — Spark's
    # own overwrite can't clear it through the macOS FUSE mount.
    shutil.rmtree(tmp_path, ignore_errors=True)
    df_out.write.mode("overwrite").parquet(tmp_path)
    shutil.rmtree(out_path, ignore_errors=True)
    os.rename(tmp_path, out_path)
    open(marker, "w").close()
    print(f"[gold_label] saved: {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_label") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_gold_label(
        args.snapshotdate,
        "datamart/silver/lms/",
        "datamart/gold/label/",
        spark,
    )

    spark.stop()
