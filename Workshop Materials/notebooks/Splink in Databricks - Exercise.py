# Databricks notebook source
# MAGIC %md
# MAGIC # Splink on Databricks - Exercise
# MAGIC 
# MAGIC ***
# MAGIC 
# MAGIC This notebook was tested on a cluster running the DBR 12.1 ML

# COMMAND ----------

# MAGIC %pip install splink --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Libraries

# COMMAND ----------

from splink.spark.spark_linker import SparkLinker
import splink.spark.spark_comparison_library as cl


import pandas as pd

from IPython.display import IFrame

from utils.splink_linker_model import SplinkLinkerModel
from utils.mlflow_utils import *
from utils.mlflow import splink_mlflow

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC Set parameters based on active user

# COMMAND ----------

username = spark.sql('select current_user() as user').collect()[0]['user']
db_name = f"{username.replace('.', '_').replace('@', '_')}_splink_data"
db_name

# COMMAND ----------

table_name = "cleansed_company_officers"
data = spark.read.table(db_name+'.'+table_name)

# COMMAND ----------

# DBTITLE 1,Remove cached Splink tables
#Splink has an issue where changes to the input data are not reflected in it's cached tables. Run this step after doing any feature engineering.
x = spark.sql(f"show tables from {db_name} like '*__splink__*'").collect()
for _ in x:
  spark.sql(f"drop table {db_name}.{_.tableName}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building a baseline model
# MAGIC 
# MAGIC Lastly, we'll build a baseline model (using essentially the set-up of the previous demonstration), and log it as the first run in an MLFlow experiment.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We'll set an experiment based on the user ID generated at the start. You can navigate to it via
# MAGIC 
# MAGIC **Left sidebar --> Workspace --> Shared --> \<your-user-name\>_databricks_experiment**

# COMMAND ----------

mlflow.set_experiment(f"/Shared/{db_name}_experiment")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parameters
# MAGIC 
# MAGIC Let's set the parameters as in the demonstration notebook.

# COMMAND ----------

blocking_rules_to_generate_predictions = [
  "l.name = r.name and l.date_of_birth = r.date_of_birth",
  "l.nationality = r.nationality and l.locality = r.locality and l.premises = r.premises and l.region = r.region",
  "l.address_line_1 = r.address_line_1 and l.postal_code = r.postal_code and l.surname = r.surname",
]

comparisons = [
  cl.levenshtein_at_thresholds("surname", 10),
  cl.levenshtein_at_thresholds("forenames", 10),
  cl.levenshtein_at_thresholds("address_line_1", 5),
  cl.levenshtein_at_thresholds("country_of_residence", 1),
  cl.levenshtein_at_thresholds("nationality", 2)
]

deterministic_rules = [
  "l.name = r.name and levenshtein(r.date_of_birth, l.date_of_birth) <= 1",
  "l.address_line_1 = r.address_line_1 and levenshtein(l.name, r.name) <= 5",
  "l.name = r.name and levenshtein(l.address_line_1, r.address_line_1) <= 5",
]

settings = {
  "retain_intermediate_calculation_columns": True,
  "retain_matching_columns": True,
  "link_type": "dedupe_only",
  "unique_id_column_name": "uid",
  "comparisons": comparisons,
  "em_convergence": 0.01,
  "blocking_rules_to_generate_predictions": blocking_rules_to_generate_predictions
}

training_rules = [
  "l.name = r.name and l.date_of_birth = r.date_of_birth",
  "l.surname = r.surname and l.forenames = r.forenames"
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Runnig a trial
# MAGIC 
# MAGIC Let's do an initial run with the above parameters. Notice that this time we're making predictions inside the run as well to also log our loss and the plot generated inside the run.
# MAGIC 
# MAGIC The run should appear in your experiment (you may need to refresh if you have the tab open), or you can navigate to it via the hyperlink provided after the cell has started running.

# COMMAND ----------

# Start MLflow run
with mlflow.start_run() as run:
  # Retrieve run ID
  RUN_ID = run.info.run_id

  # Train linker model
  linker = SparkLinker(data, spark=spark, database=db_name)
  linker.initialise_settings(settings)
  linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.8)
  linker.estimate_u_using_random_sampling(target_rows=1e7)
  for training_rule in training_rules:
    linker.estimate_parameters_using_expectation_maximisation(training_rule)
  
  # Log model and parameters
  splink_mlflow.log_splink_model_to_mlflow(linker, "linker")

  # Make predictions
  predictions = linker.predict()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Splink gives us the likelihood that a pair of records are the same, and also the clusters that are created by following the chain of pairs. If A&B are the same, and B&C are the same, then A&B&C are the same. This is a cluster.

# COMMAND ----------

# pairwise predictions
predictions.as_pandas_dataframe(limit=10)

# COMMAND ----------

# Creating the clusters requires setting a confidence threshold to indicate the degree of similarity between pairs. 
# If A&B have a 90% chance of the being the same, and B&C a 10% chance, we don't want to connect A&B&C.

clusters = linker.cluster_pairwise_predictions_at_threshold(predictions, threshold_match_probability=0.5)
clusters.as_pandas_dataframe(limit=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using the trained model
# MAGIC 
# MAGIC In the above MLflow run, we created and trained a `model` object. We can use this object directly for further predictions, or to make use of Splink's many built-in exploratory tools.

# COMMAND ----------

linker.m_u_parameters_chart()

# COMMAND ----------

path=f"/Users/{username}/clusters.html"


linker.cluster_studio_dashboard(predictions, clusters, path, sampling_method="by_cluster_size", overwrite=True)

# Splink writes to dbfs, so we need to prepend our path with /dbfs
with open("/dbfs"+path, "r") as f:
    html2=f.read()
    
displayHTML(html2)

# COMMAND ----------

path=f"/Users/{username}/scv.html"

linker.comparison_viewer_dashboard(predictions, path, overwrite=True)

# Splink writes to dbfs, so we need to prepend our path with /dbfs
with open("/dbfs"+path, "r") as f:
    html=f.read()
    
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC To visualise clusters together, you can use the `visualise_linked_records` for a static (but less readable) picture of the connected records.
# MAGIC 
# MAGIC Use the `linkage_threshold` argument to set the minimum match probability, and the `min_linked_records` argument to set a minimum number of records in a cluster for it to be visualised. Avoid setting these too low or the visualisation will be unreadable.

# COMMAND ----------

visualise_linked_records(predictions.as_pandas_dataframe(), min_linked_records=10, linkage_threshold=0.9)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Turn
# MAGIC 
# MAGIC You now have a baseline model which no doubt could use some improvement, which will be up to you to make.
# MAGIC 
# MAGIC Suggestions for work:
# MAGIC 1. Explore the data with Spark/Pandas to better understand the contents of the columns - use Splink's built in profilers to help you.
# MAGIC 2. Use your intuition, your understanding of the data and how the distance functions work to devise a new set or sets of parameters that make more sense given the task.
# MAGIC 3. Train new models and use Splink's built-in visualisers to explore how your predictions were made.
# MAGIC 4. Calculate the loss and compare it against the baseline or previous runs.
# MAGIC 5. Submit new runs to your experiment. Track how the loss evolves with your different trials on the experiments page.
# MAGIC 6. Try and de-duplicate your data based on the clusters that you find.
# MAGIC 6. **If you need any help at all, reach out to a TA!**
# MAGIC 7. Prepare 1 slide for show and tell at the end of the hack.
