# Databricks notebook source
# MAGIC %pip install --quiet splink # mlflow hyperopt

# COMMAND ----------

from arc.autolinker.autolinker import AutoLinker

# COMMAND ----------

# MAGIC %md
# MAGIC # Read Data
# MAGIC Data downloaded and sampled from https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution

# COMMAND ----------

# DBTITLE 1,Load the data and verify its size.
from pyspark.sql.functions import monotonically_increasing_id
data = spark.read.table("marcell_autosplink.voters_data").withColumn('uid', monotonically_increasing_id())
data.count()

# COMMAND ----------

data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Auto-Linking

# COMMAND ----------

# DBTITLE 1,Create new linker object
autolinker = AutoLinker(
  catalog = "robert_whiffin_uc", schema="autosplink" # optional arguments to determine where internal tables are stored
)

# COMMAND ----------

# DBTITLE 1,Set autolinking settings
autolinker.auto_link(
  data=data,                                                             # dataset to dedupe
  attribute_columns=["givenname", "surname", "suburb", "postcode"],      # columns that contain attributes to compare
  unique_id="uid",                                                       # column name of the unique ID
  comparison_size_limit=300e6,                                          # Maximum number of pairs when blocking applied
  max_evals=1                                                            # Maximum number of hyperopt trials to run
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

