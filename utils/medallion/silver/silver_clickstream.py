import os
import argparse

import pyspark
from pyspark.sql.functions import col
from pyspark.sql.types import DateType


def process_silver_clickstream(snapshot_date_str, bronze_cs_dir, silver_cs_dir, spark):
    filepath = os.path.join(bronze_cs_dir, f"bronze_clickstream_{snapshot_date_str.replace('-', '_')}.csv")
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver_clickstream] loaded: {filepath} | rows: {df.count()}")

    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    os.makedirs(silver_cs_dir, exist_ok=True)
    out_path = os.path.join(silver_cs_dir, "silver_clickstream.parquet")
    partition_path = os.path.join(out_path, f"snapshot_date={snapshot_date_str}")

    if os.path.exists(partition_path):
        print(f"[silver_clickstream] {snapshot_date_str} already exists, skipping.")
        return

    # append this month; dynamic partition overwrite prevents duplicate rows on re-run
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    df.write.mode("overwrite").partitionBy("snapshot_date").parquet(out_path)
    print(f"[silver_clickstream] saved: {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_clickstream") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_silver_clickstream(
        args.snapshotdate,
        "datamart/bronze/clickstream/",
        "datamart/silver/clickstream/",
        spark,
    )

    spark.stop()
