# Databricks notebook source
# MAGIC %pip install --quiet splink mlflow hyperopt

# COMMAND ----------

from arc.autolinker.autolinker import AutoLinker

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
  , experiment_name="/Users/robert.whiffin@databricks.com/voter_data_benchmark_photon_updated_loss_groupby_dedupe"
)

# COMMAND ----------

# DBTITLE 1,Set autolinking settings
autolinker.auto_link(
  data=voter_data,                                                             # dataset to dedupe
  attribute_columns=["givenname", "surname", "suburb", "postcode"],      # columns that contain attributes to compare
  unique_id="uid",                                                       # column name of the unique ID
  max_evals=50,                                                            # Maximum number of trials to run
  true_id="recid"
)

# COMMAND ----------

display(
  autolinker.best_predictions()
)

# COMMAND ----------

(autolinker.best_predictions().count(), voter_data.count())

# COMMAND ----------

autolinker.best_clusters_at_threshold().count() == voter_data.count()

# COMMAND ----------

autolinker.cluster_viewer()

# COMMAND ----------

autolinker.comparison_viewer()

# COMMAND ----------

autolinker.match_weights_chart()

