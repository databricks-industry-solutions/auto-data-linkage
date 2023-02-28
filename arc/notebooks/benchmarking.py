# Databricks notebook source
# MAGIC %pip install --quiet splink mlflow hyperopt

# COMMAND ----------

from arc.autolinker.autolinker import AutoLinker
from arc.autolinker.empirical_evaluations import calculate_empirical_score

# COMMAND ----------

# MAGIC %md
# MAGIC # Read Data
# MAGIC Data downloaded and sampled from https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution

# COMMAND ----------

# DBTITLE 1,Load the data and verify its size.
voter_data = spark.read.table("marcell_autosplink.voters_data_sample")

# COMMAND ----------

voter_data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Auto-Linking

# COMMAND ----------

# DBTITLE 1,Create new linker object
autolinker = AutoLinker(
  catalog = "robert_whiffin_uc", schema="autosplink" # optional arguments to determine where internal tables are stored
  , experiment_name="/Users/robert.whiffin@databricks.com/voter_data_benchmark_photon"
)

# COMMAND ----------

# DBTITLE 1,Set autolinking settings
autolinker.auto_link(
  data=voter_data,                                                             # dataset to dedupe
  attribute_columns=["givenname", "surname", "suburb", "postcode"],      # columns that contain attributes to compare
  unique_id="uid",                                                       # column name of the unique ID
  max_evals=600,                                                            # Maximum number of trials to run
  true_id="recid"
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

