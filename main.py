import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.bronze.bronze_lms
import utils.bronze.bronze_financials
import utils.bronze.bronze_attributes
import utils.bronze.bronze_clickstream
import utils.silver.silver_lms
import utils.silver.silver_fin_attr
import utils.silver.silver_clickstream
import utils.gold.gold_label
import utils.gold.gold_features
import utils.gold.gold_join


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

####  bronze table setup

# set up bronze datalake directories
bronze_root_dir = "datamart/bronze/"
bronze_lms_dir = bronze_root_dir + "lms/"
bronze_financials_dir = bronze_root_dir
bronze_attributes_dir = bronze_root_dir
bronze_clickstream_dir = bronze_root_dir

for d in [bronze_root_dir, bronze_lms_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# create bronze datalake for lms (partitioning by snapshot date)
print(f"\n{'='*60}")
print(f"[BRONZE] Starting bronze table: lms ({len(dates_str_lst)} partitions: {dates_str_lst[0]} to {dates_str_lst[-1]})")
print(f"{'='*60}")
lms_start = datetime.now()
df_bronze_lms = None
for date_str in dates_str_lst:
    print(f"  [lms] Processing partition: {date_str} ...")
    df_bronze_lms = utils.bronze.bronze_lms.process_bronze_lms_table(date_str, bronze_lms_dir, spark)
    print(f"  [lms] Done partition: {date_str} | rows: {df_bronze_lms.count()} | cols: {len(df_bronze_lms.columns)}")
lms_elapsed = (datetime.now() - lms_start).total_seconds()
print(f"[BRONZE] Done bronze table: lms | partitions: {len(dates_str_lst)} | elapsed: {lms_elapsed:.1f}s")

# create bronze datalake for financials, attributes and clickstream features
bronze_dfs = {}
for table_name, process_fn, table_dir in [
    ("financials",   utils.bronze.bronze_financials.process_bronze_financials_table,   bronze_financials_dir),
    ("attributes",   utils.bronze.bronze_attributes.process_bronze_attributes_table,   bronze_attributes_dir),
    ("clickstream",  utils.bronze.bronze_clickstream.process_bronze_clickstream_table, bronze_clickstream_dir),
]:
    print(f"\n{'='*60}")
    print(f"[BRONZE] Starting bronze table: {table_name}")
    print(f"{'='*60}")
    t_start = datetime.now()
    bronze_dfs[table_name] = process_fn(table_dir, spark)
    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"[BRONZE] Done bronze table: {table_name} | rows: {bronze_dfs[table_name].count()} | cols: {len(bronze_dfs[table_name].columns)} | elapsed: {elapsed:.1f}s")

####  silver table setup

# set up silver datalake directories
silver_root_dir = "datamart/silver/"
silver_lms_dir = silver_root_dir + "loan_daily/"
silver_fin_attr_dir = silver_root_dir 
silver_clickstream_dir = silver_root_dir

for d in [silver_lms_dir, silver_fin_attr_dir, silver_clickstream_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# run silver backfill for lms data
print(f"\n{'='*60}")
print(f"[SILVER] Starting silver table: lms ({len(dates_str_lst)} partitions: {dates_str_lst[0]} to {dates_str_lst[-1]})")
print(f"{'='*60}")
df_silver_lms = None
for date_str in dates_str_lst:
    df_silver_lms = utils.silver.silver_lms.process_silver_lms_table(date_str, bronze_lms_dir, silver_lms_dir, spark)

print(f"\n{'='*60}")
print("[SILVER] Starting silver table: fin_attr")
print(f"{'='*60}")
df_silver_financials_attributes = utils.silver.silver_fin_attr.process_silver_fin_attr_join(bronze_root_dir, silver_fin_attr_dir, spark)

print(f"\n{'='*60}")
print("[SILVER] Starting silver table: clickstream")
print(f"{'='*60}")
df_silver_clickstream = utils.silver.silver_clickstream.process_silver_clickstream_table(bronze_root_dir, silver_clickstream_dir, spark)

# preview first 5 rows of each silver table
print(f"\n{'='*60}")
print("[SILVER] Preview: lms (last partition)")
print(f"{'='*60}")
df_silver_lms.show(5, truncate=False)

print(f"\n{'='*60}")
print("[SILVER] Preview: financials_attributes")
print(f"{'='*60}")
df_silver_financials_attributes.show(5, truncate=False)

print(f"\n{'='*60}")
print("[SILVER] Preview: clickstream")
print(f"{'='*60}")
df_silver_clickstream.show(5, truncate=False)

####  gold table setup

# set up gold datalake directories
gold_root_dir = "datamart/gold/"
gold_features_dir = gold_root_dir
gold_label_dir = gold_root_dir + "label_store/"

for d in [gold_root_dir, gold_label_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# run gold features
print(f"\n{'='*60}")
print("[GOLD] Starting gold table: features")
print(f"{'='*60}")
t_start = datetime.now()
df_gold_features = utils.gold.gold_features.process_gold_features_join(silver_fin_attr_dir, silver_clickstream_dir, gold_features_dir, spark)
elapsed = (datetime.now() - t_start).total_seconds()
print(f"[GOLD] Done gold table: features | rows: {df_gold_features.count()} | cols: {len(df_gold_features.columns)} | elapsed: {elapsed:.1f}s")

# run gold label backfill
print(f"\n{'='*60}")
print(f"[GOLD] Starting gold table: label ({len(dates_str_lst)} partitions: {dates_str_lst[0]} to {dates_str_lst[-1]})")
print(f"{'='*60}")
t_start = datetime.now()
df_gold_label = None
for date_str in dates_str_lst:
    df_gold_label = utils.gold.gold_label.process_labels_gold_table(date_str, silver_lms_dir, gold_label_dir, spark, dpd=30, mob=6)
elapsed = (datetime.now() - t_start).total_seconds()
print(f"[GOLD] Done gold table: label | partitions: {len(dates_str_lst)} | elapsed: {elapsed:.1f}s")

# run gold features-label join
print(f"\n{'='*60}")
print("[GOLD] Starting gold table: features_label_join")
print(f"{'='*60}")
t_start = datetime.now()
df_gold_join = utils.gold.gold_join.process_gold_join(gold_features_dir, gold_label_dir, gold_root_dir, spark)
elapsed = (datetime.now() - t_start).total_seconds()
print(f"[GOLD] Done gold table: features_label_join | rows: {df_gold_join.count()} | cols: {len(df_gold_join.columns)} | elapsed: {elapsed:.1f}s")

# preview first 5 rows of each gold table
print(f"\n{'='*60}")
print("[GOLD] Preview: features")
print(f"{'='*60}")
df_gold_features.show(5, truncate=False)

print(f"\n{'='*60}")
print("[GOLD] Preview: label (last partition)")
print(f"{'='*60}")
df_gold_label.show(5, truncate=False)

print(f"\n{'='*60}")
print("[GOLD] Preview: features_label_join")
print(f"{'='*60}")
df_gold_join.show(5, truncate=False)

# print schema for all bronze, silver and gold tables
def print_schema(table_name, df):
    print(f"\n{table_name}:")
    for field in df.schema.fields:
        print(f"  {field.name:<45} {field.dataType.simpleString()}")
    print(f"  Shape: ({df.count()} rows, {len(df.columns)} cols)")

print(f"\n{'='*60}")
print("SCHEMAS")
print(f"{'='*60}")

print_schema("bronze_lms", df_bronze_lms)
print(f"  Partitions: {len(dates_str_lst)} | {dates_str_lst[0]} to {dates_str_lst[-1]}")
for name, df in bronze_dfs.items():
    print_schema(f"bronze_{name}", df)

print_schema("silver_lms", df_silver_lms)
print(f"  Partitions: {len(dates_str_lst)} | {dates_str_lst[0]} to {dates_str_lst[-1]}")
print_schema("silver_financials_attributes", df_silver_financials_attributes)
print_schema("silver_clickstream", df_silver_clickstream)

print_schema("gold_features", df_gold_features)
print_schema("gold_label (one partition)", df_gold_label)
print(f"  Partitions: {len(dates_str_lst)} | {dates_str_lst[0]} to {dates_str_lst[-1]}")
print_schema("features_label_join", df_gold_join)
    