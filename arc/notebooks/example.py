# Databricks notebook source
# MAGIC %pip install --quiet splink # mlflow hyperopt

# COMMAND ----------

from arc.autolinker.autolinker import AutoLinker

# COMMAND ----------

# MAGIC %md
# MAGIC # Read Data
# MAGIC Data downloaded and sampled from https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution

# COMMAND ----------

data = spark.read.table("marcell_autosplink.voters_data_sample")

# COMMAND ----------

# this is here to set defaults - Autolinker() will use the spark default catalog and database if not provided as args
spark.sql("use catalog robert_whiffin_uc")
spark.sql("use database autosplink")

# COMMAND ----------

data.display()


# COMMAND ----------

# MAGIC %md
# MAGIC # Run Auto-Linking

# COMMAND ----------

autolinker = AutoLinker()

# COMMAND ----------

autolinker.auto_link(
  data=data,                                                             # dataset to dedupe
  attribute_columns=["givenname", "surname", "suburb", "postcode"],      # columns that contain attributes to compare
  unique_id="uid",                                                       # column name of the unique ID
  comparison_size_limit=200000,                                          # Maximum number of pairs when blocking applied
  max_evals=10                                                            # Maximum number of hyperopt trials to run
)

# COMMAND ----------

display(
  autolinker.best_predictions()
)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import Window

display(
  autolinker.best_clusters_at_threshold()# default=0.8
  .withColumn("size", F.count("*").over(Window.partitionBy("cluster_id")))
  .orderBy(-F.col("size"))
)

# COMMAND ----------

autolinker.cluster_viewer()

# COMMAND ----------

autolinker.comparison_viewer()

# COMMAND ----------

autolinker.match_weights_chart()

