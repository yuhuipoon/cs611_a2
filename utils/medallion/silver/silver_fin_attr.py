import os
import argparse

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_attr(snapshot_date_str, bronze_attr_dir, spark):
    filepath = os.path.join(bronze_attr_dir, f"bronze_attributes_{snapshot_date_str.replace('-', '_')}.csv")
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver_attr] loaded: {filepath} | rows: {df.count()}")

    total = df.count()

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))

    # Name — drop PII
    df = df.drop("Name")

    # Age — strip trailing underscores, null if out of range [18, 56]
    null_before = df.filter(col("Age").isNull()).count()
    df = df.withColumn("Age", F.regexp_replace(col("Age").cast(StringType()), r"_", ""))
    df = df.withColumn("Age", col("Age").cast(IntegerType()))
    df = df.withColumn("Age", F.when((col("Age") >= 18) & (col("Age") <= 56), col("Age")).otherwise(None).cast(IntegerType()))
    print(f"  Age: {df.filter(col('Age').isNull()).count() - null_before} / {total} nulled (out of range).")

    # SSN — replace non-conforming with 'Unknown'
    ssn_pattern = r'^\d{3}-\d{2}-\d{4}$'
    df = df.withColumn("SSN", col("SSN").cast(StringType()))
    non_conforming = df.filter(~col("SSN").rlike(ssn_pattern) | col("SSN").isNull()).count()
    df = df.withColumn("SSN", F.when(col("SSN").rlike(ssn_pattern), col("SSN")).otherwise("Unknown"))
    print(f"  SSN: {non_conforming} / {total} replaced with 'Unknown'.")

    # Occupation — replace symbols-only values with 'Unknown'
    symbols_only = r'^[^a-zA-Z]+$'
    df = df.withColumn("Occupation", col("Occupation").cast(StringType()))
    non_conforming = df.filter(col("Occupation").rlike(symbols_only) | col("Occupation").isNull()).count()
    df = df.withColumn("Occupation", F.when(col("Occupation").rlike(symbols_only), "Unknown").otherwise(col("Occupation")))
    print(f"  Occupation: {non_conforming} / {total} replaced with 'Unknown'.")

    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    return df


def process_silver_fin(snapshot_date_str, bronze_fin_dir, spark):
    filepath = os.path.join(bronze_fin_dir, f"bronze_financials_{snapshot_date_str.replace('-', '_')}.csv")
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver_fin] loaded: {filepath} | rows: {df.count()}")

    total = df.count()

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))

    # Annual_Income
    df = df.withColumn("Annual_Income", F.regexp_replace(col("Annual_Income").cast(StringType()), r"_", ""))
    df = df.withColumn("Annual_Income", F.round(col("Annual_Income").cast(FloatType()), 2))

    # Monthly_Inhand_Salary
    df = df.withColumn("Monthly_Inhand_Salary", F.round(col("Monthly_Inhand_Salary").cast(FloatType()), 2))

    # Income_Inconsistent flag
    df = df.withColumn("Income_Inconsistent", (col("Monthly_Inhand_Salary") * 12 > col("Annual_Income")))

    # Num_Bank_Accounts — null if > 11
    null_before = df.filter(col("Num_Bank_Accounts").isNull()).count()
    df = df.withColumn("Num_Bank_Accounts", F.when(col("Num_Bank_Accounts").cast(IntegerType()) <= 11, col("Num_Bank_Accounts")).otherwise(None).cast(IntegerType()))
    print(f"  Num_Bank_Accounts: {df.filter(col('Num_Bank_Accounts').isNull()).count() - null_before} / {total} nulled (> 11).")

    # Num_Credit_Card — null if > 11
    null_before = df.filter(col("Num_Credit_Card").isNull()).count()
    df = df.withColumn("Num_Credit_Card", F.when(col("Num_Credit_Card").cast(IntegerType()) <= 11, col("Num_Credit_Card")).otherwise(None).cast(IntegerType()))
    print(f"  Num_Credit_Card: {df.filter(col('Num_Credit_Card').isNull()).count() - null_before} / {total} nulled (> 11).")

    # Interest_Rate — null if > 34
    null_before = df.filter(col("Interest_Rate").isNull()).count()
    df = df.withColumn("Interest_Rate", F.when(col("Interest_Rate").cast(IntegerType()) <= 34, col("Interest_Rate")).otherwise(None).cast(IntegerType()))
    print(f"  Interest_Rate: {df.filter(col('Interest_Rate').isNull()).count() - null_before} / {total} nulled (> 34).")

    # Num_of_Loan — infer from Type_of_Loan comma count
    df = df.withColumn("_inferred_num_loans",
        F.when(col("Type_of_Loan").isNull(), 0)
         .otherwise(F.size(F.split(col("Type_of_Loan"), ",")))
         .cast(IntegerType()))
    df = df.withColumn("Num_of_Loan", col("_inferred_num_loans")).drop("_inferred_num_loans")

    # Type_of_Loan — null → 'Unknown'
    df = df.withColumn("Type_of_Loan", F.when(col("Type_of_Loan").isNull(), "Unknown").otherwise(col("Type_of_Loan")))

    # Delay_from_due_date — null if negative or > 62
    null_before = df.filter(col("Delay_from_due_date").isNull()).count()
    df = df.withColumn("Delay_from_due_date",
        F.when((col("Delay_from_due_date").cast(IntegerType()) >= 0) & (col("Delay_from_due_date").cast(IntegerType()) <= 62), col("Delay_from_due_date"))
         .otherwise(None).cast(IntegerType()))
    print(f"  Delay_from_due_date: {df.filter(col('Delay_from_due_date').isNull()).count() - null_before} / {total} nulled (out of range).")

    # Num_of_Delayed_Payment — strip underscores, null if negative or > 28
    null_before = df.filter(col("Num_of_Delayed_Payment").isNull()).count()
    df = df.withColumn("Num_of_Delayed_Payment", F.regexp_replace(col("Num_of_Delayed_Payment").cast(StringType()), r"_", ""))
    df = df.withColumn("Num_of_Delayed_Payment",
        F.when((col("Num_of_Delayed_Payment").cast(IntegerType()) >= 0) & (col("Num_of_Delayed_Payment").cast(IntegerType()) <= 28), col("Num_of_Delayed_Payment"))
         .otherwise(None).cast(IntegerType()))
    print(f"  Num_of_Delayed_Payment: {df.filter(col('Num_of_Delayed_Payment').isNull()).count() - null_before} / {total} nulled (out of range).")

    # Delay_Inconsistent flag
    df = df.withColumn("Delay_Inconsistent",
        F.when(
            ((col("Delay_from_due_date") > 0) & (col("Num_of_Delayed_Payment") == 0)) |
            ((col("Delay_from_due_date") == 0) & (col("Num_of_Delayed_Payment") > 0)),
            True
        ).otherwise(False))

    # Changed_Credit_Limit — strip underscores, empty → null
    null_before = df.filter(col("Changed_Credit_Limit").isNull()).count()
    df = df.withColumn("Changed_Credit_Limit", F.regexp_replace(col("Changed_Credit_Limit").cast(StringType()), r"_", ""))
    df = df.withColumn("Changed_Credit_Limit", F.when(col("Changed_Credit_Limit") == "", None).otherwise(col("Changed_Credit_Limit")).cast(FloatType()))
    print(f"  Changed_Credit_Limit: {df.filter(col('Changed_Credit_Limit').isNull()).count() - null_before} / {total} nulled ('_' values).")

    # Num_Credit_Inquiries — null if > 17
    null_before = df.filter(col("Num_Credit_Inquiries").isNull()).count()
    df = df.withColumn("Num_Credit_Inquiries",
        F.when(col("Num_Credit_Inquiries").cast(IntegerType()) <= 17, col("Num_Credit_Inquiries"))
         .otherwise(None).cast(IntegerType()))
    print(f"  Num_Credit_Inquiries: {df.filter(col('Num_Credit_Inquiries').isNull()).count() - null_before} / {total} nulled (> 17).")

    # Credit_Mix — '_' or null → 'Unknown'
    non_conforming = df.filter((col("Credit_Mix") == "_") | col("Credit_Mix").isNull()).count()
    df = df.withColumn("Credit_Mix",
        F.when((col("Credit_Mix") == "_") | col("Credit_Mix").isNull(), "Unknown")
         .otherwise(col("Credit_Mix")).cast(StringType()))
    print(f"  Credit_Mix: {non_conforming} / {total} replaced with 'Unknown'.")

    # Outstanding_Debt
    df = df.withColumn("Outstanding_Debt", F.regexp_replace(col("Outstanding_Debt").cast(StringType()), r"_", ""))
    df = df.withColumn("Outstanding_Debt", F.round(col("Outstanding_Debt").cast(FloatType()), 2))

    # Credit_Utilization_Ratio
    df = df.withColumn("Credit_Utilization_Ratio", F.round(col("Credit_Utilization_Ratio").cast(FloatType()), 2))

    # Credit_History_Age — keep as string; also parse to months
    df = df.withColumn("Credit_History_Age", col("Credit_History_Age").cast(StringType()))
    df = df.withColumn("Credit_History_Age_Months",
        F.when(col("Credit_History_Age").isNull(), None)
         .otherwise(
             F.coalesce(F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+[Yy]ear", 1).cast(IntegerType()), F.lit(0)) * 12 +
             F.coalesce(F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+[Mm]onth", 1).cast(IntegerType()), F.lit(0))
         ).cast(IntegerType()))

    # Payment_of_Min_Amount
    df = df.withColumn("Payment_of_Min_Amount", col("Payment_of_Min_Amount").cast(StringType()))

    # Total_EMI_per_month — null if > 863
    null_before = df.filter(col("Total_EMI_per_month").isNull()).count()
    df = df.withColumn("Total_EMI_per_month",
        F.when(col("Total_EMI_per_month").cast(FloatType()) <= 863, F.round(col("Total_EMI_per_month").cast(FloatType()), 2))
         .otherwise(None))
    print(f"  Total_EMI_per_month: {df.filter(col('Total_EMI_per_month').isNull()).count() - null_before} / {total} nulled (> 863).")

    # Amount_Invested_Monthly — sentinel → null
    null_before = df.filter(col("Amount_Invested_Monthly").isNull()).count()
    df = df.withColumn("Amount_Invested_Monthly",
        F.when(col("Amount_Invested_Monthly").cast(StringType()) == "__10000__", None)
         .otherwise(F.round(col("Amount_Invested_Monthly").cast(FloatType()), 2)))
    print(f"  Amount_Invested_Monthly: {df.filter(col('Amount_Invested_Monthly').isNull()).count() - null_before} / {total} nulled (sentinel).")

    # Payment_Behaviour — extract Spent_Level and Value_Payments
    df = df.withColumn("Payment_Behaviour",
        F.when(col("Payment_Behaviour") == "!@9#%8", "Unknown_spent_Unknown_value_payments")
         .otherwise(col("Payment_Behaviour")).cast(StringType()))
    df = df.withColumn("Spent_Level", F.regexp_extract(col("Payment_Behaviour"), r"^(.+)_spent_", 1).cast(StringType()))
    df = df.withColumn("Value_Payments", F.regexp_extract(col("Payment_Behaviour"), r"_spent_(.+)_value_payments$", 1).cast(StringType()))
    df = df.drop("Payment_Behaviour")

    # Monthly_Balance — sentinel → null
    null_before = df.filter(col("Monthly_Balance").isNull()).count()
    df = df.withColumn("Monthly_Balance",
        F.when(col("Monthly_Balance").cast(StringType()) == "__-333333333333333333333333333__", None)
         .otherwise(F.round(col("Monthly_Balance").cast(FloatType()), 2)))
    print(f"  Monthly_Balance: {df.filter(col('Monthly_Balance').isNull()).count() - null_before} / {total} nulled (sentinel).")

    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    df = df.select([
        "Customer_ID", "Annual_Income", "Monthly_Inhand_Salary", "Income_Inconsistent",
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Type_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Delay_Inconsistent",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix",
        "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Credit_History_Age", "Credit_History_Age_Months",
        "Payment_of_Min_Amount", "Total_EMI_per_month", "Amount_Invested_Monthly",
        "Spent_Level", "Value_Payments", "Monthly_Balance", "snapshot_date",
    ])

    return df


def process_silver_fin_attr(snapshot_date_str, bronze_fin_dir, bronze_attr_dir, silver_fin_attr_dir, spark):
    df_attr = process_silver_attr(snapshot_date_str, bronze_attr_dir, spark)
    df_fin = process_silver_fin(snapshot_date_str, bronze_fin_dir, spark)

    df = df_attr.join(df_fin, on=["Customer_ID", "snapshot_date"], how="left")

    attr_cols = df_attr.columns
    fin_cols = [c for c in df_fin.columns if c not in ("Customer_ID", "snapshot_date")]
    df = df.select(attr_cols + fin_cols)
    print(f"[silver_fin_attr] joined: {df.count()} rows, {len(df.columns)} cols.")

    os.makedirs(silver_fin_attr_dir, exist_ok=True)
    out_path = os.path.join(silver_fin_attr_dir, "silver_fin_attr.parquet")
    partition_path = os.path.join(out_path, f"snapshot_date={snapshot_date_str}")

    if os.path.exists(partition_path):
        print(f"[silver_fin_attr] {snapshot_date_str} already exists, skipping.")
        return

    # append this month; dynamic partition overwrite prevents duplicate rows on re-run
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    df.write.mode("overwrite").partitionBy("snapshot_date").parquet(out_path)
    print(f"[silver_fin_attr] saved: {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_fin_attr") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_silver_fin_attr(
        args.snapshotdate,
        "datamart/bronze/financials/",
        "datamart/bronze/attributes/",
        "datamart/silver/fin_attr/",
        spark,
    )

    spark.stop()
