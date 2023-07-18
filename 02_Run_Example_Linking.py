# Databricks notebook source
# MAGIC %md
# MAGIC # Install ARC and Splink

# COMMAND ----------

# MAGIC %pip install --quiet databricks-arc splink

# COMMAND ----------

# MAGIC %md
# MAGIC # Load ARC

# COMMAND ----------

from arc.autolinker import AutoLinker
import arc

# COMMAND ----------

# MAGIC %md
# MAGIC # Enable ARC

# COMMAND ----------

arc.enable_arc()

# COMMAND ----------

# MAGIC %md
# MAGIC # Read test data

# COMMAND ----------
import os
data = spark.read.csv(f"file:{os.getcwd()}/data/arc_febrl1.csv", header=True)

data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Perform Auto-linking

# COMMAND ----------

autolinker = AutoLinker()

attribute_columns = ["given_name", "surname", "street_number", "address_1", "address_2", "suburb", "postcode", "state", "date_of_birth"]

autolinker.auto_link(
  data=data,                                                         # Spark DataFrame of data to deduplicate
  attribute_columns=attribute_columns,                               # List of column names containing attribute to compare
  unique_id="uid",                                                   # Name of the unique identifier column
  comparison_size_limit=100000,                                      # Maximum number of pairs to compare
  max_evals=10                                                       # Number of trials to run during optimisation process
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Get clusters from best model

# COMMAND ----------

clusters = autolinker.best_clusters_at_threshold(threshold=0.3)
clusters.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieve pairwise predictions

# COMMAND ----------

autolinker.best_predictions_df.as_spark_dataframe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Use Splink functionality

# COMMAND ----------

autolinker.cluster_viewer()

# COMMAND ----------

autolinker.match_weights_chart()

# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieve the best model directly from `autolinker`

# COMMAND ----------

linker = autolinker.best_linker

# COMMAND ----------

linker.profile_columns("surname")
