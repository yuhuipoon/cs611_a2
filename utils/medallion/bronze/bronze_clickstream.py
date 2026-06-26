import os
import argparse

import pyspark
from pyspark.sql.functions import col
from datetime import datetime


def process_bronze_clickstream(snapshot_date_str, bronze_cs_dir, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # static input is baked into the image at /data (no FUSE bind mount, so
    # Spark reads no longer hit EDEADLK on macOS Docker Desktop)
    df = spark.read.csv("/data/feature_clickstream.csv", header=True, inferSchema=True) \
               .filter(col("snapshot_date") == snapshot_date)

    os.makedirs(bronze_cs_dir, exist_ok=True)
    partition_name = f"bronze_clickstream_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_cs_dir, partition_name)

    if os.path.exists(filepath):
        print(f"[bronze_clickstream] {snapshot_date_str} already exists, skipping.")
        return

    print(f"[bronze_clickstream] {snapshot_date_str} | rows: {df.count()}")
    df.toPandas().to_csv(filepath, index=False)
    print(f"[bronze_clickstream] saved: {filepath}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("bronze_clickstream") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_bronze_clickstream(args.snapshotdate, "datamart/bronze/clickstream/", spark)

    spark.stop()
