import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_attr_table(bronze_dir, spark):
    filepath = bronze_dir + "bronze_attributes.csv"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    total = df.count()

    # Customer_ID
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))

    # Name — drop PII
    df = df.drop("Name")

    # Age
    null_before = df.filter(col("Age").isNull()).count()
    df = df.withColumn("Age", F.regexp_replace(col("Age").cast(StringType()), r"_", ""))
    df = df.withColumn("Age", col("Age").cast(IntegerType()))
    df = df.withColumn("Age", F.when((col("Age") >= 18) & (col("Age") <= 56), col("Age")).otherwise(None).cast(IntegerType()))
    null_after = df.filter(col("Age").isNull()).count()
    print(f"  Age: {null_after - null_before} / {total} nulled (out of range).")

    # SSN
    ssn_pattern = r'^\d{3}-\d{2}-\d{4}$'
    df = df.withColumn("SSN", col("SSN").cast(StringType()))
    non_conforming = df.filter(~col("SSN").rlike(ssn_pattern) | col("SSN").isNull()).count()
    df = df.withColumn("SSN", F.when(col("SSN").rlike(ssn_pattern), col("SSN")).otherwise("Unknown"))
    print(f"  SSN: {non_conforming} / {total} replaced with 'Unknown'.")

    # Occupation
    symbols_only = r'^[^a-zA-Z]+$'
    df = df.withColumn("Occupation", col("Occupation").cast(StringType()))
    non_conforming = df.filter(col("Occupation").rlike(symbols_only) | col("Occupation").isNull()).count()
    df = df.withColumn("Occupation", F.when(col("Occupation").rlike(symbols_only), "Unknown").otherwise(col("Occupation")))
    print(f"  Occupation: {non_conforming} / {total} replaced with 'Unknown'.")

    # snapshot_date
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    print(f"  Attributes processed: {df.count()} rows, {len(df.columns)} cols.")
    return df


def process_silver_fin_table(bronze_dir, spark):
    filepath = bronze_dir + "bronze_financials.csv"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Customer_ID
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))

    # Annual_Income — strip trailing _, cast to float, round to 2dp
    df = df.withColumn("Annual_Income", F.regexp_replace(col("Annual_Income").cast(StringType()), r"_", ""))
    df = df.withColumn("Annual_Income", F.round(col("Annual_Income").cast(FloatType()), 2))

    # Monthly_Inhand_Salary — round to 2dp, cast to float
    df = df.withColumn("Monthly_Inhand_Salary", F.round(col("Monthly_Inhand_Salary").cast(FloatType()), 2))

    # Income_Inconsistent — flag where annualised monthly salary exceeds reported annual income
    df = df.withColumn("Income_Inconsistent", (col("Monthly_Inhand_Salary") * 12 > col("Annual_Income")))

    # Num_Bank_Accounts — null if > 11, cast to int
    total = df.count()
    null_before = df.filter(col("Num_Bank_Accounts").isNull()).count()
    df = df.withColumn("Num_Bank_Accounts", F.when(col("Num_Bank_Accounts").cast(IntegerType()) <= 11, col("Num_Bank_Accounts")).otherwise(None).cast(IntegerType()))
    null_after = df.filter(col("Num_Bank_Accounts").isNull()).count()
    print(f"  Num_Bank_Accounts: {null_after - null_before} / {total} nulled (> 11).")

    # Num_Credit_Card — null if > 11, cast to int
    null_before = df.filter(col("Num_Credit_Card").isNull()).count()
    df = df.withColumn("Num_Credit_Card", F.when(col("Num_Credit_Card").cast(IntegerType()) <= 11, col("Num_Credit_Card")).otherwise(None).cast(IntegerType()))
    null_after = df.filter(col("Num_Credit_Card").isNull()).count()
    print(f"  Num_Credit_Card: {null_after - null_before} / {total} nulled (> 11).")

    # Interest_Rate — null if > 34, cast to int
    null_before = df.filter(col("Interest_Rate").isNull()).count()
    df = df.withColumn("Interest_Rate", F.when(col("Interest_Rate").cast(IntegerType()) <= 34, col("Interest_Rate")).otherwise(None).cast(IntegerType()))
    null_after = df.filter(col("Interest_Rate").isNull()).count()
    print(f"  Interest_Rate: {null_after - null_before} / {total} nulled (> 34).")

    # Num_of_Loan — infer from Type_of_Loan comma count before null replacement; null → 0 loans
    df = df.withColumn("_inferred_num_loans",
        F.when(col("Type_of_Loan").isNull(), 0)
         .otherwise(F.size(F.split(col("Type_of_Loan"), ",")))
         .cast(IntegerType()))

    mismatches_df = df.filter(
        col("Num_of_Loan").isNull() |
        (col("_inferred_num_loans") != col("Num_of_Loan").cast(IntegerType()))
    )
    mismatch_count = mismatches_df.count()
    print(f"  Num_of_Loan: {mismatch_count} / {total} inferred counts mismatched original.")

    mismatch_dist = (
        mismatches_df
        .groupBy(
            col("Num_of_Loan").cast(IntegerType()).alias("original"),
            col("_inferred_num_loans").alias("inferred")
        )
        .count()
        .orderBy("original", "inferred")
    )
    for row in mismatch_dist.collect():
        print(f"    {row['original']} --> {row['inferred']}: {row['count']} rows")

    df = df.withColumn("Num_of_Loan", col("_inferred_num_loans")).drop("_inferred_num_loans")
    print(f"  Num_of_Loan: replaced with inferred count, cast to int.")


    # Type_of_Loan — replace null with "Unknown"
    null_before = df.filter(col("Type_of_Loan").isNull()).count()
    df = df.withColumn("Type_of_Loan", F.when(col("Type_of_Loan").isNull(), "Unknown").otherwise(col("Type_of_Loan")))
    print(f"  Type_of_Loan: {null_before} / {total} nulls replaced with 'Unknown'.")

    # Delay_from_due_date — null if negative or > 62, cast to int
    null_before = df.filter(col("Delay_from_due_date").isNull()).count()
    df = df.withColumn("Delay_from_due_date",
        F.when((col("Delay_from_due_date").cast(IntegerType()) >= 0) & (col("Delay_from_due_date").cast(IntegerType()) <= 62), col("Delay_from_due_date"))
         .otherwise(None).cast(IntegerType()))
    null_after = df.filter(col("Delay_from_due_date").isNull()).count()
    print(f"  Delay_from_due_date: {null_after - null_before} / {total} nulled (negative or > 62).")

    # Num_of_Delayed_Payment — strip trailing _, null if negative or > 28, cast to int
    null_before = df.filter(col("Num_of_Delayed_Payment").isNull()).count()
    df = df.withColumn("Num_of_Delayed_Payment", F.regexp_replace(col("Num_of_Delayed_Payment").cast(StringType()), r"_", ""))
    df = df.withColumn("Num_of_Delayed_Payment",
        F.when((col("Num_of_Delayed_Payment").cast(IntegerType()) >= 0) & (col("Num_of_Delayed_Payment").cast(IntegerType()) <= 28), col("Num_of_Delayed_Payment"))
         .otherwise(None).cast(IntegerType()))
    null_after = df.filter(col("Num_of_Delayed_Payment").isNull()).count()
    print(f"  Num_of_Delayed_Payment: {null_after - null_before} / {total} nulled (negative or > 28).")

    # Delay_Inconsistent — True if delay days > 0 but payment count = 0, or vice versa
    df = df.withColumn("Delay_Inconsistent",
        F.when(
            ((col("Delay_from_due_date") > 0) & (col("Num_of_Delayed_Payment") == 0)) |
            ((col("Delay_from_due_date") == 0) & (col("Num_of_Delayed_Payment") > 0)),
            True
        ).otherwise(False))

    # Changed_Credit_Limit — strip underscores; empty result → null, cast to float
    null_before = df.filter(col("Changed_Credit_Limit").isNull()).count()
    df = df.withColumn("Changed_Credit_Limit", F.regexp_replace(col("Changed_Credit_Limit").cast(StringType()), r"_", ""))
    df = df.withColumn("Changed_Credit_Limit", F.when(col("Changed_Credit_Limit") == "", None).otherwise(col("Changed_Credit_Limit")).cast(FloatType()))
    null_after = df.filter(col("Changed_Credit_Limit").isNull()).count()
    print(f"  Changed_Credit_Limit: {null_after - null_before} / {total} nulled ('_' values).")

    # Num_Credit_Inquiries — null if > 17, cast to int
    null_before = df.filter(col("Num_Credit_Inquiries").isNull()).count()
    df = df.withColumn("Num_Credit_Inquiries",
        F.when(col("Num_Credit_Inquiries").cast(IntegerType()) <= 17, col("Num_Credit_Inquiries"))
         .otherwise(None).cast(IntegerType()))
    null_after = df.filter(col("Num_Credit_Inquiries").isNull()).count()
    print(f"  Num_Credit_Inquiries: {null_after - null_before} / {total} nulled (> 17).")

    # Credit_Mix — replace "_" with "Unknown", cast to string
    non_conforming = df.filter((col("Credit_Mix") == "_") | col("Credit_Mix").isNull()).count()
    df = df.withColumn("Credit_Mix",
        F.when((col("Credit_Mix") == "_") | col("Credit_Mix").isNull(), "Unknown")
         .otherwise(col("Credit_Mix")).cast(StringType()))
    print(f"  Credit_Mix: {non_conforming} / {total} replaced with 'Unknown'.")

    # Outstanding_Debt — strip trailing _, round to 2dp, cast to float
    df = df.withColumn("Outstanding_Debt", F.regexp_replace(col("Outstanding_Debt").cast(StringType()), r"_", ""))
    df = df.withColumn("Outstanding_Debt", F.round(col("Outstanding_Debt").cast(FloatType()), 2))

    # Credit_Utilization_Ratio — round to 2dp, cast to float
    df = df.withColumn("Credit_Utilization_Ratio", F.round(col("Credit_Utilization_Ratio").cast(FloatType()), 2))

    # Credit_History_Age — keep as string
    df = df.withColumn("Credit_History_Age", col("Credit_History_Age").cast(StringType()))

    # Credit_History_Age_Months — parse "X Years and Y Months" into total months, cast to int
    df = df.withColumn("Credit_History_Age_Months",
        F.when(col("Credit_History_Age").isNull(), None)
         .otherwise(
             F.coalesce(F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+[Yy]ear", 1).cast(IntegerType()), F.lit(0)) * 12 +
             F.coalesce(F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+[Mm]onth", 1).cast(IntegerType()), F.lit(0))
         ).cast(IntegerType()))

    # Payment_of_Min_Amount — keep as string
    df = df.withColumn("Payment_of_Min_Amount", col("Payment_of_Min_Amount").cast(StringType()))

    # Total_EMI_per_month — null if > 863, round to 2dp, cast to float
    null_before = df.filter(col("Total_EMI_per_month").isNull()).count()
    df = df.withColumn("Total_EMI_per_month",
        F.when(col("Total_EMI_per_month").cast(FloatType()) <= 863, F.round(col("Total_EMI_per_month").cast(FloatType()), 2))
         .otherwise(None))
    null_after = df.filter(col("Total_EMI_per_month").isNull()).count()
    print(f"  Total_EMI_per_month: {null_after - null_before} / {total} nulled (> 863).")

    # Amount_Invested_Monthly — replace sentinel "__10000__" with null, round to 2dp, cast to float
    null_before = df.filter(col("Amount_Invested_Monthly").isNull()).count()
    df = df.withColumn("Amount_Invested_Monthly",
        F.when(col("Amount_Invested_Monthly").cast(StringType()) == "__10000__", None)
         .otherwise(F.round(col("Amount_Invested_Monthly").cast(FloatType()), 2)))
    null_after = df.filter(col("Amount_Invested_Monthly").isNull()).count()
    print(f"  Amount_Invested_Monthly: {null_after - null_before} / {total} nulled ('__10000__' sentinel).")

    # Payment_Behaviour — replace sentinel, extract Spent_Level and Value_Payments, drop original
    non_conforming = df.filter(col("Payment_Behaviour") == "!@9#%8").count()
    df = df.withColumn("Payment_Behaviour",
        F.when(col("Payment_Behaviour") == "!@9#%8", "Unknown_spent_Unknown_value_payments")
         .otherwise(col("Payment_Behaviour")).cast(StringType()))
    df = df.withColumn("Spent_Level",
        F.regexp_extract(col("Payment_Behaviour"), r"^(.+)_spent_", 1).cast(StringType()))
    df = df.withColumn("Value_Payments",
        F.regexp_extract(col("Payment_Behaviour"), r"_spent_(.+)_value_payments$", 1).cast(StringType()))
    df = df.drop("Payment_Behaviour")
    print(f"  Payment_Behaviour: {non_conforming} / {total} replaced with 'Unknown_spent_Unknown_value_payments'.")

    # Monthly_Balance — replace sentinel with null, round to 2dp, cast to float
    null_before = df.filter(col("Monthly_Balance").isNull()).count()
    df = df.withColumn("Monthly_Balance",
        F.when(col("Monthly_Balance").cast(StringType()) == "__-333333333333333333333333333__", None)
         .otherwise(F.round(col("Monthly_Balance").cast(FloatType()), 2)))
    null_after = df.filter(col("Monthly_Balance").isNull()).count()
    print(f"  Monthly_Balance: {null_after - null_before} / {total} nulled (sentinel).")

    # snapshot_date — cast to date
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # enforce column order
    df = df.select([
        "Customer_ID",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Income_Inconsistent",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Type_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Delay_Inconsistent",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Credit_Mix",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Credit_History_Age",
        "Credit_History_Age_Months",
        "Payment_of_Min_Amount",
        "Total_EMI_per_month",
        "Amount_Invested_Monthly",
        "Spent_Level",
        "Value_Payments",
        "Monthly_Balance",
        "snapshot_date",
    ])

    print(f"  Financials processed: {df.count()} rows, {len(df.columns)} cols.")
    return df


def process_silver_fin_attr_join(bronze_dir, silver_fin_attr_directory, spark):
    df_attr = process_silver_attr_table(bronze_dir, spark)
    df_fin = process_silver_fin_table(bronze_dir, spark)

    # left join on Customer_ID and snapshot_date — single snapshot_date in result
    df = df_attr.join(df_fin, on=["Customer_ID", "snapshot_date"], how="left")

    # reorder: attr columns first (original order), then fin columns (excluding Customer_ID and snapshot_date)
    attr_cols = df_attr.columns
    fin_cols = [c for c in df_fin.columns if c not in ("Customer_ID", "snapshot_date")]
    df = df.select(attr_cols + fin_cols)
    print(f"  Joined table: {df.count()} rows, {len(df.columns)} cols.")

    output_filepath = silver_fin_attr_directory + "silver_financials_attributes.parquet"
    df.write.mode("overwrite").parquet(output_filepath)
    print('saved to:', output_filepath)

    return df