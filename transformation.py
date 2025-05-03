# Databricks notebook source
# MAGIC %md
# MAGIC ##start session
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Peerisland").getOrCreate()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the tranformation info
# MAGIC

# COMMAND ----------

import logging
import os

log_directory = '/Volumes/peerisland/peer/logs'
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(log_directory, 'transformation.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read the customer data

# COMMAND ----------

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.functions import *

initial_df = spark.read.option('header', 'true')\
    .option('inferschema',True)\
    .csv("/Volumes/peerisland/peer/peer_vol/customers-100000.csv")
                    
display(initial_df)



# COMMAND ----------

# MAGIC %md
# MAGIC ###Tranformation Funtion
# MAGIC

# COMMAND ----------

#tranformation functions

def with_column(df: DataFrame, column_name: str, expr: str):
    logging.info(f"Adding column '{column_name}' with expression '{expr}'")
    return df.withColumn(column_name, expr)

def drop_columns(df: DataFrame, columns: list):
    logging.info(f"Dropping columns: {', '.join(columns)}")
    return df.drop(*columns)


def cf_concat_columns(df: DataFrame, output_col: str = "Full_Name", input_cols: list = None):
    logging.info(f"Concatenating columns: {input_cols} into '{output_col}'")
    return df.withColumn(output_col, F.concat_ws(' ', *[df[col] for col in input_cols]))


def cf_add_primary_key(df: DataFrame, key_column_name: str = "Customer_ID", columns_to_hash: list = None):
    logging.info(f"Applying custom transformation: add primary key column '{key_column_name}'")
    if columns_to_hash is None:
        columns_to_hash = df.columns
    concatenated_cols = F.concat_ws('||', *[df[col] for col in columns_to_hash])
    return df.withColumn(key_column_name, F.sha2(concatenated_cols, 256))

def cf_extract_year_month(df: DataFrame, date_column: str):
    """Extract year and month from date column"""
    logging.info(f"Extracting year and month from column '{date_column}'")
    return (df
            .withColumn('yearofSubscribe', year(col(date_column)))
            .withColumn('monthofsubscribe', month(col(date_column)))
           )


def mask_email(df: DataFrame, email_column: str):
    logging.info(f"Applying custom transformation: mask email in column '{email_column}'")
    return df.withColumn('Email', F.regexp_replace(df[email_column], r'(?<=.{2}).(?=.*@)', '*'))  

def mask_phone(df: DataFrame, phone_column: str):
    logging.info(f"Applying custom transformation: mask phone in column '{phone_column}'")
    return df.withColumn( phone_column,F.concat(lit("******"),substring(F.col(phone_column), -5, 5))
    )


def cf_replace_null(df: DataFrame):
    logging.info("Applying custom transformation: replace null values")
    if df.filter(F.reduce(lambda a, b: a | b, (df[col].isNull() for col in df.columns))).count() > 0:
        return df.fillna('Unknown')
    else:
        logging.info("No null values found in the file") 
    return df

def cf_drop_duplicates(df: DataFrame, *columns: str):
    if not columns:
        if df.count() == df.dropDuplicates().count():
            logging.info("No duplicates found in the DataFrame")
        return df.dropDuplicates()
    if df.count() == df.dropDuplicates(columns).count():
        logging.info("No duplicates found in the specified columns")
    return df.dropDuplicates(columns)

def cf_add_load_time(df: DataFrame):
    logging.info("Applying custom transformation: add load_time")
    return df.withColumn('load_time', F.current_timestamp())





# COMMAND ----------

# MAGIC %md
# MAGIC ####Transformation Dictionary
# MAGIC

# COMMAND ----------


TRANSFORMATIONS = {
    'with_column': with_column,
    'drop_columns': drop_columns,
    'concat_columns': cf_concat_columns,
    'add_primary_key': cf_add_primary_key,
    'extract_year_month': cf_extract_year_month,
    'mask_email': mask_email,
    'mask_phone': mask_phone,
    'replace_null': cf_replace_null,
    'drop_duplicates': cf_drop_duplicates,
    'add_load_time': cf_add_load_time
}

# COMMAND ----------

# MAGIC %md
# MAGIC ###Function to apply transformation

# COMMAND ----------


def apply_transformations(df: DataFrame, operations: list):
    for transformation in operations:
        operation = transformation.get('operation')
        params = transformation.get('params', {})

        if operation not in TRANSFORMATIONS:
            logging.error(f"Operation '{operation}' not recognized.")
            continue 

        try:
            
            df = TRANSFORMATIONS[operation](df, **params)
        except Exception as e:
            logging.error(f"Error applying transformation '{operation}': {str(e)}")
            continue 

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transforamtion config to apply on DF

# COMMAND ----------

transformations_config = [
    {'operation': 'with_column', 'params': {'column_name': 'Phone_num1', 'expr': regexp_replace(col("Phone 1"), r"[^\d+]", "")}},
    {'operation': 'with_column', 'params': {'column_name': 'Phone_num2', 'expr': regexp_replace(col("Phone 2"), r"[^\d+]", "")}},
    {'operation': 'concat_columns', 'params': {'output_col': 'Full_Name', 'input_cols': ['First Name', 'Last Name']}},
    {'operation': 'drop_columns', 'params': {'columns': ['First Name', 'Last Name']}},
    {'operation': 'add_primary_key', 'params': {'key_column_name': 'Customer_ID', 'columns_to_hash': ['Index', 'Customer Id']}},
    {'operation': 'drop_columns', 'params': {'columns': ['Index', 'Customer Id']}},
     {'operation': 'extract_year_month', 'params': {'date_column': 'Subscription Date'}},
    {'operation': 'mask_email', 'params': {'email_column': 'Email'}},
    {'operation': 'mask_phone', 'params': {'phone_column': 'Phone_num1'}},
    {'operation': 'mask_phone', 'params': {'phone_column': 'Phone_num2'}},
    {'operation': 'drop_columns', 'params': {'columns': ['Phone 1', 'Phone 2']}},
    {'operation': 'replace_null', 'params': {}},
    {'operation': 'drop_duplicates', 'params': {}},
    {'operation': 'add_load_time', 'params': {}}
]


# COMMAND ----------

# MAGIC %md
# MAGIC ####Applying tarnsformation of DF
# MAGIC

# COMMAND ----------

final_df = apply_transformations(initial_df, transformations_config)

display(final_df)

