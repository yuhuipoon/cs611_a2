import pandas as pd
import pyspark.sql.functions as F


def process_bronze_clickstream_table(bronze_clickstream_directory, spark):
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/feature_clickstream.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    print('row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    filepath = bronze_clickstream_directory + "bronze_clickstream.csv"
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df