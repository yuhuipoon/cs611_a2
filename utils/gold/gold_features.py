import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType

def process_gold_fin_attr(silver_fin_attr_dir, spark):
    filepath = silver_fin_attr_dir + "silver_financials_attributes.parquet"
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    total = df.count()

    # Customer_ID — keep as is

    # Age — impute nulls with median
    age_median = df.select(F.percentile_approx("Age", 0.5)).first()[0]
    null_before = df.filter(col("Age").isNull()).count()
    df = df.withColumn("Age", F.when(col("Age").isNull(), age_median).otherwise(col("Age")).cast(IntegerType()))
    print(f"  Age: {null_before} / {total} nulls imputed with median ({age_median}).")

    # SSN — drop
    df = df.drop("SSN")
    print(f"  SSN: dropped.")

    # Occupation — one-hot encode
    occupations = sorted([row["Occupation"] for row in df.select("Occupation").distinct().collect()])
    for occ in occupations:
        df = df.withColumn(f"Occupation_{occ}", F.when(col("Occupation") == occ, 1).otherwise(0).cast(IntegerType()))
    df = df.drop("Occupation")
    print(f"  Occupation: one-hot encoded into {len(occupations)} columns: {occupations}.")

    # Annual_Income — drop
    df = df.drop("Annual_Income")
    print(f"  Annual_Income: dropped.")

    # Monthly_Inhand_Salary — keep as is
    # Monthly_Inhand_Salary_binned — bin into 5 ranges
    df = df.withColumn("Monthly_Inhand_Salary_binned",
        F.when(col("Monthly_Inhand_Salary") < 1500, "<1500")
         .when(col("Monthly_Inhand_Salary") < 2600, "1500-2600")
         .when(col("Monthly_Inhand_Salary") < 4000, "2600-4000")
         .when(col("Monthly_Inhand_Salary") < 6700, "4000-6700")
         .otherwise(">6700").cast(StringType()))
    bin_order = ["<1500", "1500-2600", "2600-4000", "4000-6700", ">6700"]
    bin_counts = {row["Monthly_Inhand_Salary_binned"]: row["count"] for row in df.groupBy("Monthly_Inhand_Salary_binned").count().collect()}
    print(f"  Monthly_Inhand_Salary_binned:")
    for b in bin_order:
        print(f"    {b:<15} {bin_counts.get(b, 0)}")

    # Income_Inconsistent — drop
    df = df.drop("Income_Inconsistent")
    print(f"  Income_Inconsistent: dropped.")

    # Num_Bank_Accounts — impute nulls with median
    nba_median = df.select(F.percentile_approx("Num_Bank_Accounts", 0.5)).first()[0]
    null_before = df.filter(col("Num_Bank_Accounts").isNull()).count()
    df = df.withColumn("Num_Bank_Accounts", F.when(col("Num_Bank_Accounts").isNull(), nba_median).otherwise(col("Num_Bank_Accounts")).cast(IntegerType()))
    print(f"  Num_Bank_Accounts: {null_before} / {total} nulls imputed with median ({nba_median}).")

    # Num_Credit_Card — impute nulls with median
    ncc_median = df.select(F.percentile_approx("Num_Credit_Card", 0.5)).first()[0]
    null_before = df.filter(col("Num_Credit_Card").isNull()).count()
    df = df.withColumn("Num_Credit_Card", F.when(col("Num_Credit_Card").isNull(), ncc_median).otherwise(col("Num_Credit_Card")).cast(IntegerType()))
    print(f"  Num_Credit_Card: {null_before} / {total} nulls imputed with median ({ncc_median}).")

    # Interest_Rate — impute nulls with median
    ir_median = df.select(F.percentile_approx("Interest_Rate", 0.5)).first()[0]
    null_before = df.filter(col("Interest_Rate").isNull()).count()
    df = df.withColumn("Interest_Rate", F.when(col("Interest_Rate").isNull(), ir_median).otherwise(col("Interest_Rate")).cast(IntegerType()))
    print(f"  Interest_Rate: {null_before} / {total} nulls imputed with median ({ir_median}).")

    # Num_of_Loan — keep as is

    # Type_of_Loan — multi-label one-hot encode (comma-separated values; strip leading "and " from last items)
    loan_types = sorted(set([
        row["loan_type"][4:] if row["loan_type"].startswith("and ") else row["loan_type"] for row in
        df.select(F.explode(F.split(col("Type_of_Loan"), ", ")).alias("loan_type")).distinct().collect()
        if not (row["loan_type"].startswith("and ") and row["loan_type"][4:] == "Unknown"
                or row["loan_type"] == "Unknown")
    ]))
    for lt in loan_types:
        col_name = "Loan_" + lt.replace(" ", "_").replace("-", "_")
        df = df.withColumn(col_name, F.when(col("Type_of_Loan").contains(lt), 1).otherwise(0).cast(IntegerType()))
    df = df.drop("Type_of_Loan")
    loan_col_names = ["Loan_" + lt.replace(" ", "_").replace("-", "_") for lt in loan_types]
    loan_counts = {cn: df.filter(col(cn) == 1).count() for cn in loan_col_names}
    print(f"  Type_of_Loan: one-hot encoded into {len(loan_types)} columns:")
    for lt, cn in zip(loan_types, loan_col_names):
        print(f"    {cn:<45} (from '{lt}') count: {loan_counts[cn]}")

    # Delay_from_due_date — impute nulls with median
    dfdd_median = df.select(F.percentile_approx("Delay_from_due_date", 0.5)).first()[0]
    null_before = df.filter(col("Delay_from_due_date").isNull()).count()
    df = df.withColumn("Delay_from_due_date", F.when(col("Delay_from_due_date").isNull(), dfdd_median).otherwise(col("Delay_from_due_date")).cast(IntegerType()))
    print(f"  Delay_from_due_date: {null_before} / {total} nulls imputed with median ({dfdd_median}).")

    # Num_of_Delayed_Payment — impute nulls with median
    nodp_median = df.select(F.percentile_approx("Num_of_Delayed_Payment", 0.5)).first()[0]
    null_before = df.filter(col("Num_of_Delayed_Payment").isNull()).count()
    df = df.withColumn("Num_of_Delayed_Payment", F.when(col("Num_of_Delayed_Payment").isNull(), nodp_median).otherwise(col("Num_of_Delayed_Payment")).cast(IntegerType()))
    print(f"  Num_of_Delayed_Payment: {null_before} / {total} nulls imputed with median ({nodp_median}).")

    # Delay_Inconsistent — boolean to int (1 = True, 0 = False)
    df = df.withColumn("Delay_Inconsistent", col("Delay_Inconsistent").cast(IntegerType()))

    # Changed_Credit_Limit — impute nulls with median
    ccl_median = df.select(F.percentile_approx("Changed_Credit_Limit", 0.5)).first()[0]
    null_before = df.filter(col("Changed_Credit_Limit").isNull()).count()
    df = df.withColumn("Changed_Credit_Limit", F.when(col("Changed_Credit_Limit").isNull(), ccl_median).otherwise(col("Changed_Credit_Limit")).cast(FloatType()))
    print(f"  Changed_Credit_Limit: {null_before} / {total} nulls imputed with median ({ccl_median}).")

    # Num_Credit_Inquiries — impute nulls with median
    nci_median = df.select(F.percentile_approx("Num_Credit_Inquiries", 0.5)).first()[0]
    null_before = df.filter(col("Num_Credit_Inquiries").isNull()).count()
    df = df.withColumn("Num_Credit_Inquiries", F.when(col("Num_Credit_Inquiries").isNull(), nci_median).otherwise(col("Num_Credit_Inquiries")).cast(IntegerType()))
    print(f"  Num_Credit_Inquiries: {null_before} / {total} nulls imputed with median ({nci_median}).")

    # Credit_Mix — ordinal encode: Bad=0, Unknown=1, Standard=2, Good=3
    df = df.withColumn("Credit_Mix",
        F.when(col("Credit_Mix") == "Bad", 0)
         .when(col("Credit_Mix") == "Unknown", 1)
         .when(col("Credit_Mix") == "Standard", 2)
         .when(col("Credit_Mix") == "Good", 3)
         .otherwise(None).cast(IntegerType()))

    # Credit_Mix_Known — 0 if Unknown, 1 if Bad/Standard/Good
    df = df.withColumn("Credit_Mix_Known",
        F.when(col("Credit_Mix") == 1, 0).otherwise(1).cast(IntegerType()))

    # Outstanding_Debt — keep as is
    # Outstanding_Debt_binned — bin into 3 ranges
    df = df.withColumn("Outstanding_Debt_binned",
        F.when(col("Outstanding_Debt") < 1500, "<1500")
         .when(col("Outstanding_Debt") < 2500, "1500-2500")
         .otherwise(">2500").cast(StringType()))
    bin_order = ["<1500", "1500-2500", ">2500"]
    bin_counts = {row["Outstanding_Debt_binned"]: row["count"] for row in df.groupBy("Outstanding_Debt_binned").count().collect()}
    print(f"  Outstanding_Debt_binned:")
    for b in bin_order:
        print(f"    {b:<15} {bin_counts.get(b, 0)}")

    # Credit_Utilization_Ratio — keep as is

    # Credit_History_Age — drop
    df = df.drop("Credit_History_Age")
    print(f"  Credit_History_Age: dropped.")

    # Credit_History_Age_Months — keep as is

    # Payment_of_Min_Amount — one-hot encode
    poma_values = sorted([row["Payment_of_Min_Amount"] for row in df.select("Payment_of_Min_Amount").distinct().collect()])
    for v in poma_values:
        df = df.withColumn(f"Payment_of_Min_Amount_{v}", F.when(col("Payment_of_Min_Amount") == v, 1).otherwise(0).cast(IntegerType()))
    df = df.drop("Payment_of_Min_Amount")
    poma_col_names = [f"Payment_of_Min_Amount_{v}" for v in poma_values]
    poma_counts = {cn: df.filter(col(cn) == 1).count() for cn in poma_col_names}
    print(f"  Payment_of_Min_Amount: one-hot encoded into {len(poma_values)} columns:")
    for v, cn in zip(poma_values, poma_col_names):
        print(f"    {cn:<45} count: {poma_counts[cn]}")

    # Total_EMI_per_month — impute nulls with median
    temi_median = df.select(F.percentile_approx("Total_EMI_per_month", 0.5)).first()[0]
    null_before = df.filter(col("Total_EMI_per_month").isNull()).count()
    df = df.withColumn("Total_EMI_per_month", F.when(col("Total_EMI_per_month").isNull(), temi_median).otherwise(col("Total_EMI_per_month")).cast(FloatType()))
    print(f"  Total_EMI_per_month: {null_before} / {total} nulls imputed with median ({temi_median}).")

    # Total_EMI_per_month_binned — bin into 5 ranges
    df = df.withColumn("Total_EMI_per_month_binned",
        F.when(col("Total_EMI_per_month") < 25, "<25")
         .when(col("Total_EMI_per_month") < 50, "25-50")
         .when(col("Total_EMI_per_month") < 100, "50-100")
         .when(col("Total_EMI_per_month") < 200, "100-200")
         .otherwise(">200").cast(StringType()))
    bin_order = ["<25", "25-50", "50-100", "100-200", ">200"]
    bin_counts = {row["Total_EMI_per_month_binned"]: row["count"] for row in df.groupBy("Total_EMI_per_month_binned").count().collect()}
    print(f"  Total_EMI_per_month_binned:")
    for b in bin_order:
        print(f"    {b:<15} {bin_counts.get(b, 0)}")

    # Amount_Invested_Monthly — impute nulls with median
    aim_median = df.select(F.percentile_approx("Amount_Invested_Monthly", 0.5)).first()[0]
    null_before = df.filter(col("Amount_Invested_Monthly").isNull()).count()
    df = df.withColumn("Amount_Invested_Monthly", F.when(col("Amount_Invested_Monthly").isNull(), aim_median).otherwise(col("Amount_Invested_Monthly")).cast(FloatType()))
    print(f"  Amount_Invested_Monthly: {null_before} / {total} nulls imputed with median ({aim_median}).")

    # Amount_Invested_Monthly_binned — bin into 5 ranges
    df = df.withColumn("Amount_Invested_Monthly_binned",
        F.when(col("Amount_Invested_Monthly") < 65, "<65")
         .when(col("Amount_Invested_Monthly") < 110, "65-110")
         .when(col("Amount_Invested_Monthly") < 170, "110-170")
         .when(col("Amount_Invested_Monthly") < 320, "170-320")
         .otherwise(">320").cast(StringType()))
    bin_order = ["<65", "65-110", "110-170", "170-320", ">320"]
    bin_counts = {row["Amount_Invested_Monthly_binned"]: row["count"] for row in df.groupBy("Amount_Invested_Monthly_binned").count().collect()}
    print(f"  Amount_Invested_Monthly_binned:")
    for b in bin_order:
        print(f"    {b:<15} {bin_counts.get(b, 0)}")

    # Spent_Level — replace "Unknown" via proportional random sampling, then ordinal encode (Low=0, High=1)
    import random as _random
    known_counts = {row["Spent_Level"]: row["count"] for row in df.filter(col("Spent_Level") != "Unknown").groupBy("Spent_Level").count().collect()}
    total_known = sum(known_counts.values())
    proportions = {k: v / total_known for k, v in known_counts.items()}
    n_unknown = df.filter(col("Spent_Level") == "Unknown").count()
    print(f"  Spent_Level: {n_unknown} / {total} 'Unknown' to be replaced proportionally {proportions}.")

    values_sl = list(proportions.keys())
    weights_sl = [proportions[v] for v in values_sl]
    bc_sl = spark.sparkContext.broadcast((values_sl, weights_sl))

    @F.udf(StringType())
    def sample_spent_level(_):
        v, w = bc_sl.value
        return _random.choices(v, weights=w, k=1)[0]

    df = df.withColumn("Spent_Level", F.when(col("Spent_Level") == "Unknown", sample_spent_level(F.lit(0))).otherwise(col("Spent_Level")))
    df = df.withColumn("Spent_Level",
        F.when(col("Spent_Level") == "Low", 0)
         .when(col("Spent_Level") == "High", 1)
         .otherwise(None).cast(IntegerType()))

    # Value_Payments — replace "Unknown" via proportional random sampling, then ordinal encode (Small=0, Medium=1, Large=2)
    known_counts = {row["Value_Payments"]: row["count"] for row in df.filter(col("Value_Payments") != "Unknown").groupBy("Value_Payments").count().collect()}
    total_known = sum(known_counts.values())
    proportions = {k: v / total_known for k, v in known_counts.items()}
    n_unknown = df.filter(col("Value_Payments") == "Unknown").count()
    print(f"  Value_Payments: {n_unknown} / {total} 'Unknown' to be replaced proportionally {proportions}.")

    values_vp = list(proportions.keys())
    weights_vp = [proportions[v] for v in values_vp]
    bc_vp = spark.sparkContext.broadcast((values_vp, weights_vp))

    @F.udf(StringType())
    def sample_value_payments(_):
        v, w = bc_vp.value
        return _random.choices(v, weights=w, k=1)[0]

    df = df.withColumn("Value_Payments", F.when(col("Value_Payments") == "Unknown", sample_value_payments(F.lit(0))).otherwise(col("Value_Payments")))
    df = df.withColumn("Value_Payments",
        F.when(col("Value_Payments") == "Small", 0)
         .when(col("Value_Payments") == "Medium", 1)
         .when(col("Value_Payments") == "Large", 2)
         .otherwise(None).cast(IntegerType()))

    # Monthly_Balance — drop (high correlation with Monthly_Inhand_Salary)
    df = df.drop("Monthly_Balance")
    print(f"  Monthly_Balance: dropped (high correlation with Monthly_Inhand_Salary).")

    # snapshot_date — keep as is

    print(f"  Gold fin_attr processed: {df.count()} rows, {len(df.columns)} cols.")
    return df


def process_gold_features_join(silver_fin_attr_dir, silver_clickstream_dir, gold_features_dir, spark):
    df_fin_attr = process_gold_fin_attr(silver_fin_attr_dir, spark)
    df_fin_attr = df_fin_attr.withColumnRenamed("snapshot_date", "fin_attr_snapshot_date")

    # load unaggregated clickstream and rename snapshot_date
    cs_filepath = silver_clickstream_dir + "silver_clickstream.parquet"
    df_clickstream = spark.read.parquet(cs_filepath)
    df_clickstream = df_clickstream.withColumnRenamed("snapshot_date", "cs_snapshot_date")
    print(f"  Loaded clickstream: {df_clickstream.count()} rows, {df_clickstream.select('Customer_ID').distinct().count()} unique customers.")

    # left join — every fin_attr row is kept; clickstream rows fan out where multiple cs dates exist
    df = df_fin_attr.join(df_clickstream, on="Customer_ID", how="left")
    df = df.withColumn("has_clickstream", F.when(col("cs_snapshot_date").isNotNull(), 1).otherwise(0).cast(IntegerType()))
    print(f"  Gold features joined: {df.count()} rows, {len(df.columns)} cols.")

    # distribution: how many cs_snapshot_date rows does each Customer_ID have (null counts as 0)
    cs_dist = (
        df.groupBy("Customer_ID")
          .agg(F.count("cs_snapshot_date").alias("num_cs_rows"))
          .groupBy("num_cs_rows")
          .agg(F.count("Customer_ID").alias("num_customers"))
          .orderBy("num_cs_rows")
    )
    print("  Distribution of cs_snapshot_date rows per Customer_ID:")
    cs_dist.show(truncate=False)

    # enforce column order
    occ_cols  = sorted([c for c in df.columns if c.startswith("Occupation_")])
    loan_cols = sorted([c for c in df.columns if c.startswith("Loan_")])
    poma_cols = sorted([c for c in df.columns if c.startswith("Payment_of_Min_Amount_")])
    fe_cols   = [f"fe_{i}" for i in range(1, 21)]

    ordered_cols = (
        ["Customer_ID", "Age"] +
        occ_cols +
        ["fin_attr_snapshot_date",
         "Monthly_Inhand_Salary", "Monthly_Inhand_Salary_binned",
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
         "Spent_Level", "Value_Payments",
         "cs_snapshot_date", "has_clickstream"] +
        fe_cols
    )
    df = df.select(ordered_cols)

    output_filepath = gold_features_dir + "gold_features.parquet"
    df.write.mode("overwrite").parquet(output_filepath)
    print('saved to:', output_filepath)

    return df
