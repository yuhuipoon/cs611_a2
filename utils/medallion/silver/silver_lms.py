import os
import argparse

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_lms(snapshot_date_str, bronze_lms_dir, silver_lms_dir, spark):
    partition_name = f"bronze_lms_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_lms_dir, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver_lms] loaded: {filepath} | rows: {df.count()}")

    # enforce schema
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    os.makedirs(silver_lms_dir, exist_ok=True)
    out_path = os.path.join(silver_lms_dir, f"silver_lms_{snapshot_date_str.replace('-', '_')}.parquet")

    if os.path.exists(out_path):
        print(f"[silver_lms] {snapshot_date_str} already exists, skipping.")
        return

    df.write.mode("overwrite").parquet(out_path)
    print(f"[silver_lms] saved: {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_lms") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_silver_lms(
        args.snapshotdate,
        "datamart/bronze/lms/",
        "datamart/silver/lms/",
        spark,
    )

    spark.stop()
