# Databricks notebook source
# MAGIC %pip install --quiet splink mlflow hyperopt

# COMMAND ----------

from arc.autolinker import AutoLinker
import arc

# COMMAND ----------

arc.sql.enable_arc()

# COMMAND ----------

# MAGIC %md
# MAGIC # Read Data
# MAGIC Data downloaded and sampled from https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution

# COMMAND ----------

# data = spark.read.table("marcell_autosplink.febrl1_uid")
data = spark.read.table("marcell_autosplink.voters_data_sample")

# COMMAND ----------

data.columns


# COMMAND ----------

# MAGIC %md
# MAGIC # Run Auto-Linking

# COMMAND ----------

autolinker = AutoLinker()

# COMMAND ----------

attr_cols = ["givenname", "surname", "postcode", "suburb"]

# COMMAND ----------

autolinker.auto_link(
  data=data,                                                             # dataset to dedupe
  attribute_columns=attr_cols,                                           # columns that contain attributes to compare
  unique_id="uid",                                                       # column name of the unique ID
  comparison_size_limit=200000,                                          # Maximum number of pairs when blocking applied
  max_evals=1000,                                                        # Maximum number of hyperopt trials to run
  threshold=0.5,
  true_label="recid"
)
