from pyspark.sql import SparkSession, DataFrame, functions as F
import logging

# logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  transformations functions

def select_columns(df: DataFrame, *columns):
    logger.info(f"Selecting columns: {columns}")
    return df.select(*columns)

def filter_data(df: DataFrame, condition: str):
    logger.info(f"Filtering with condition: {condition}")
    return df.filter(condition)

def with_column(df: DataFrame, column_name: str, expr: str):
    logger.info(f"Adding column {column_name} with expression: {expr}")
    return df.withColumn(column_name, F.expr(expr))

def drop_columns(df: DataFrame, *columns):
    logger.info(f"Dropping columns: {columns}")
    return df.drop(*columns)

def group_by(df: DataFrame, *group_columns, **agg_exprs):
    logger.info(f"Grouping by: {group_columns} with aggregation: {agg_exprs}")
    agg_exprs_list = [F.col(col).alias(agg_name) for col, agg_name in agg_exprs.items()]
    return df.groupBy(*group_columns).agg(*agg_exprs_list)

def join_dataframes(df1: DataFrame, df2: DataFrame, *on_columns, how: str = 'inner'):
    logger.info(f"Joining DataFrames on columns: {on_columns} with join type: {how}")
    return df1.join(df2, list(on_columns), how)

def custom_fun_hard_value(df: DataFrame, new_col_name: str):
    logger.info("Applying custom transformation")
    return df.withColumn(new_col_name, F.lit(1))

def apply_transformations_with_optimizations(df: DataFrame, transformations: list):
    df = df.cache()
    for transformation in transformations:
        pass
    df = df.repartition(4)
    return df

def cf_concat_columns(df: DataFrame, output_col: str = "concatenated", *input_cols: str) :
    logger.info(f"Concatenating columns: {input_cols} into '{output_col}'")
    return df.withColumn(output_col, F.concat_ws(' ', *[df[col] for col in input_cols]))


def cf_add_primary_key(df: DataFrame,key_column_name: str = "primary_key",columns_to_hash: list = None) :
    logger.info(f"Applying custom transformation: add primary key column '{key_column_name}'")
    if columns_to_hash is None:
        columns_to_hash = df.columns
    concatenated_cols = F.concat_ws('||', *[df[col] for col in columns_to_hash])
    return df.withColumn(key_column_name, F.sha2(concatenated_cols, 256))


def cf_running_total(df: DataFrame):
    logger.info("Applying custom transformation: running total")
    window_spec = Window.orderBy('timestamp').rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return df.withColumn('running_total', F.sum('amount').over(window_spec))

def cf_extract_year_month(df: DataFrame):
    logger.info("Applying custom transformation: extract year and month")
    return df.withColumn('year', F.year(df['date'])).withColumn('month', F.month(df['date']))

def cf_add_load_time(df: DataFrame):
    logger.info("Applying custom transformation: add load_time")
    return df.withColumn('load_time', F.current_timestamp())

def cf_replace_null(df: DataFrame):
    logger.info("Applying custom transformation: replace null values")
    return df.fillna({'name': 'Unknown'})

def flag_duplicates(df: DataFrame, *columns: str) -> DataFrame:
    if not columns:
        raise ValueError("At least one column must be specified to flag duplicates.")
    
    window_spec = Window.partitionBy(*columns).orderBy(*columns)
    return df.withColumn('is_duplicate', F.when(F.row_number().over(window_spec) > 1, 1).otherwise(0))

def cf_drop_duplicates(df: DataFrame, *columns: str) :
    if not columns:
        return df.dropDuplicates()
    return df.dropDuplicates(columns)


def cache_df(df: DataFrame) -> DataFrame:
    logger.info(f"[cache] Caching DataFrame")
    return df.cache()

def repartition_df(df: DataFrame, num_partitions: int) -> DataFrame:
    logger.info(f"[repartition] Repartitioning to {num_partitions} partitions")
    return df.repartition(num_partitions)    

# Transformation Dictionary 
TRANSFORMATIONS = {
    'select': select_columns,
    'filter': filter_data,
    'withColumn': with_column,
    'drop': drop_columns,
    'groupBy': group_by,
    'join': join_dataframes,
    'custom_hard_value': custom_fun_hard_value,
    'concat_columns': cf_concat_columns,
    'add_primary_key': cf_add_primary_key,
    'running_total': cf_running_total,
    'extract_year_month': cf_extract_year_month,
    'add_load_time': cf_add_load_time,
    'replace_null': cf_replace_null,
    'flag_duplicates': flag_duplicates,
    'drop_duplicates': cf_drop_duplicates,
    'repartition': repartition_df,
    'chace': cache_df
}


def apply_transformations(df: DataFrame, operations: list):
    for transformation in operations:
        operation = transformation.get('operation')
        params = transformation.get('params', {})

        if operation not in TRANSFORMATIONS:
            logger.error(f"Operation '{operation}' not recognized.")
            continue 

        try:
            
            df = TRANSFORMATIONS[operation](df, **params)
        except Exception as e:
            logger.error(f"Error applying transformation '{operation}': {str(e)}")
            continue 

    return df





#reading the df
initial_df = spark.read.csv('path_to_data.csv', header=True, inferSchema=True)

#apply list of operations on df
transformations_config = [
    {'operation': 'select', 'params': {'columns': ['name', 'age']}},
    {'operation': 'filter', 'params': {'condition': "age > 30"}},
    {'operation': 'groupBy', 'params': {'group_columns': ['country'], 'agg_exprs': {'avg_age': 'avg(age)'}}},
    {'operation': 'drop', 'params': {'columns': ['name']}},
    {'operation': 'drop_duplicates', 'params': {'columns': ['country', 'age']}},
    {'operation': 'replace_null', 'params': {}},
    {'operation': 'add_primary_key', 'params': {'key_column_name': 'primary_key'}},
    {'operation': 'extract_year_month', 'params': {}},
    {'operation': 'add_load_time', 'params': {}}
    {'operation': 'repartition', 'params': {}}
]

# applying transformtion
final_df = apply_transformations(initial_df, transformations_config)





####Action fucntions

def show_df(df: DataFrame, num_rows: int = 20) -> None:
    logger.info(f"[show] Displaying first {num_rows} rows")
    df.show(num_rows)

def collect_df(df: DataFrame) -> list:
    logger.info("[collect] Collecting DataFrame rows")
    return df.collect()

def count_rows(df: DataFrame) -> int:
    logger.info("[count] Counting rows in DataFrame")
    return df.count()

def write_to_parquet(df: DataFrame, path: str) -> None:
    logger.info(f"[write] Writing DataFrame to Parquet at {path}")
    df.write.parquet(path)

def write_to_csv(df: DataFrame, path: str) -> None:
    logger.info(f"[write] Writing DataFrame to CSV at {path}")
    df.write.csv(path)

#Action Dictionary

ACTIONS = {
    'show': show_df,
    'collect': collect_df,
    'count': count_rows,
    'write_parquet': write_to_parquet,
    'write_csv': write_to_csv
}

#to apply action

def apply_actions(df: DataFrame, action_steps: list) -> None:
    logger.info("[pipeline] Starting action pipeline...")

    for i, step in enumerate(action_steps):
        action = step.get('action')
        params = step.get('params', {})

        if action not in ACTIONS:
            raise ValueError(f"[pipeline] Unknown action '{action}' at step {i + 1}")

        logger.info(f"[pipeline] Step {i + 1}: Executing action '{action}' with params {params}")
        ACTIONS[action](df, **params)

    logger.info("[pipeline] All actions completed.")
    
    
# action steps for the pipeline
action_steps = [
    {'action': 'show', 'params': {'num_rows': 10}},  
    {'action': 'count', 'params': {}},                
    {'action': 'write_parquet', 'params': {'path': '/path/to/output.parquet'}}, 
    {'action': 'write_csv', 'params': {'path': '/path/to/output.csv'}}       
]

# applying action
apply_actions(final_df, action_steps)









