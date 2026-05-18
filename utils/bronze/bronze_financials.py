import pandas as pd
import pyspark.sql.functions as F


def process_bronze_financials_table(bronze_financials_directory, spark):
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/features_financials.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    print('row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    filepath = bronze_financials_directory + "bronze_financials.csv"
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df