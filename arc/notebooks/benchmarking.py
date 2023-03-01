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
  , experiment_name="/Users/robert.whiffin@databricks.com/voter_data_benchmark_photon_updated_dedupe_5m_limit"
)

# COMMAND ----------

import itertools
blocking_combinations = []

    # create list of all possible rules
for n in range(1, 2+1):
  combs = list(itertools.combinations(attribute_columns, n))
  blocking_combinations.extend(combs)

blocking_rules = autolinker._generate_rules(["&".join(r) for r in blocking_combinations])



comp_size_dict = dict()

for comb, rule in zip(blocking_combinations, blocking_rules):
  num_pairs = voter_data.groupBy(list(comb)).count().select((F.sum(F.col("count")*(F.col("count")-F.lit(1))))/F.lit(2)).collect()[0]["(sum((count * (count - 1))) / 2)"]
  comp_size_dict.update({rule: num_pairs})

    

# COMMAND ----------

potential_rules = []
for r in range(2, len(blocking_rules)+1):
  for c in itertools.combinations(blocking_rules, r):
    rule_size = sum([v for k, v in comp_size_dict.items() if k in c])
    potential_rules.append(list([c, rule_size]))
potential_rules

# COMMAND ----------

comparison_size_limit

# COMMAND ----------


accepted_rules = potential_rules[:4]
[_[0] for _ in accepted_rules]

# COMMAND ----------

# DBTITLE 1,Set autolinking settings
autolinker.auto_link(
  data=voter_data,                                                             # dataset to dedupe
  attribute_columns=["givenname", "surname", "suburb", "postcode"],      # columns that contain attributes to compare
  unique_id="uid",                                                       # column name of the unique ID
  max_evals=1,                                                            # Maximum number of trials to run
  true_id="recid",
  comparison_size_limit = 5e6
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

