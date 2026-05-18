import os
import glob

from pyspark.sql.functions import col


def process_gold_join(gold_features_dir, gold_label_dir, gold_joined_dir, spark):
    # load features store
    df_features = spark.read.parquet(gold_features_dir + "gold_features.parquet")
    print('loaded features from:', gold_features_dir + "gold_features.parquet", '| row count:', df_features.count())

    # load all label store partitions by globbing individual partition paths
    partition_paths = sorted(glob.glob(os.path.join(gold_label_dir, "*.parquet")))
    df_label = spark.read.parquet(*partition_paths)
    print(f'loaded labels from: {gold_label_dir} | partitions: {len(partition_paths)} | row count:', df_label.count())

    # rename snapshot_dates to distinguish source; keep features_snapshot_date for validation
    df_features = df_features.withColumnRenamed("snapshot_date", "features_snapshot_date")
    df_label = df_label.withColumnRenamed("snapshot_date", "label_snapshot_date")

    # left join on Customer_ID — keep all label records, attach features where available
    df = df_label.join(df_features, on="Customer_ID", how="left")
    print(f"  Gold joined: {df.count()} rows, {len(df.columns)} cols.")

    # check and drop rows where features were observed after the label snapshot (data leakage)
    flagged = df.filter(col("features_snapshot_date") >= col("label_snapshot_date"))
    flagged_count = flagged.count()
    if flagged_count > 0:
        print(f"  WARNING: {flagged_count} rows flagged where features_snapshot_date >= label_snapshot_date — dropping.")
    else:
        print(f"  OK: no rows with features_snapshot_date >= label_snapshot_date.")
    df = df.filter(col("features_snapshot_date") < col("label_snapshot_date"))
    df = df.drop("features_snapshot_date")
    print(f"  Gold joined after filter: {df.count()} rows, {len(df.columns)} cols.")

    # enforce column order: label cols first, then features in original order
    label_cols = ["Customer_ID", "label", "label_snapshot_date"]
    feature_cols = [c for c in df_features.columns if c not in ("Customer_ID", "features_snapshot_date")]
    df = df.select(label_cols + feature_cols)

    output_filepath = gold_joined_dir + "features_label_join.parquet"
    df.write.mode("overwrite").parquet(output_filepath)
    print('saved to:', output_filepath)

    return df
