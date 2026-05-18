def process_silver_clickstream_table(bronze_dir, silver_clickstream_directory, spark):
    # connect to bronze table
    filepath = bronze_dir + "bronze_clickstream.csv"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # save silver table - IRL connect to database to write
    output_filepath = silver_clickstream_directory + "silver_clickstream.parquet"
    df.write.mode("overwrite").parquet(output_filepath)
    print('saved to:', output_filepath)

    return df
