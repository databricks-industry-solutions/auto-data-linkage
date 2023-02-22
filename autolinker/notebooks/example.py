# Databricks notebook source
# MAGIC %pip install --quiet splink mlflow hyperopt

# COMMAND ----------

from autolinker.autolinker.autolinker import AutoLinker

# COMMAND ----------

# MAGIC %md
# MAGIC # Read Data
# MAGIC Data downloaded and sampled from https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution

# COMMAND ----------

data = spark.read.table("marcell_autosplink.voters_data_sample")

# COMMAND ----------

data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Auto-Linking

# COMMAND ----------

autolinker = AutoLinker(
  spark=spark,                                                                                            # Spark instance
  catalog="marcell_splink",                                                                               # catalog name
  schema="marcell_autosplink",                                                                            # schema to write results to
  experiment_name="/Users/marcell.ferencz@databricks.com/autosplink/evaluate/autosplink"                  # MLflow experiment location
)

# COMMAND ----------

autolinker.auto_link(
  data=data,                                                             # dataset to dedupe
  attribute_columns=["givenname", "surname", "suburb", "postcode"],      # columns that contain attributes to compare
  unique_id="uid",                                                       # column name of the unique ID
  comparison_size_limit=200000,                                          # Maximum number of pairs when blocking applied
  max_evals=1                                                            # Maximum number of hyperopt trials to run
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Test Splink functionality

# COMMAND ----------

best_linker = autolinker.best_linker

best_linker.m_u_parameters_chart()

# COMMAND ----------

predictions = autolinker.best_predictions

predictions.as_spark_dataframe().display()
