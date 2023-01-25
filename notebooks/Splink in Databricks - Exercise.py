# Databricks notebook source
# MAGIC %md
# MAGIC # Splink on Databricks - Exercise

# COMMAND ----------

#%pip install splink --quiet

# COMMAND ----------

pip install git+https://github.com/robertwhiffin/splink.git@mllfow-integration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Libraries

# COMMAND ----------

from splink.spark.spark_linker import SparkLinker
from splink.databricks.enable_splink import enable_splink
import splink.spark.spark_comparison_library as cl

import pandas as pd

from IPython.display import IFrame

from utils.splink_linker_model import SplinkLinkerModel
from utils.mlflow_utils import *

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC Set parameters based on active user

# COMMAND ----------

username = spark.sql('select current_user() as user').collect()[0]['user']
db_name = username.replace(".", "_").replace("@", "_")
db_name

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}; USE {db_name};")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluating Splink
# MAGIC 
# MAGIC Whilst Splink does offer a supervised predictive method, our data does not have ground truth (we don't categorically know which records are duplicates or otherwise), so we're resigned to taking an unsupervised approach.
# MAGIC 
# MAGIC Unsupervised models are trickier to evaluate, because we can't compare the predictions against a known truth. How would we know how well our model has performed then? How can we evaluate the impact of changing parameters against the previous experiment?
# MAGIC 
# MAGIC We are proposing a loss function that promotes match probabilities within the pairwise prediction dataframe to be close to either 1 or 0 (i.e. the model is "certain" that a pair is a match or not). For a full explanation with examples of this proposed evaluation metric, please see the notebook _Splink Evaluation - explanation_ in the same directory as this one, but for the purpose of this exercise, all we need to know is that we want to __decrease the loss__ with our predictions.
# MAGIC 
# MAGIC You can use the `get_match_probabilty_loss` method on the prediction dataframe generated by different models to compare them. The lower the loss, the better the model.
# MAGIC 
# MAGIC ***
# MAGIC 
# MAGIC ##### A word of caution
# MAGIC 
# MAGIC Please be aware that this is not a hard-and-fast, empirical way of evaluating a Splink model. Unsupervised entity resolution is complex and choosing the right model will depend on the data, use case, domain, expert knowledge, risk appetite, and many more factors.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Enabling Splink
# MAGIC 
# MAGIC As before, let's enable Splink for our notebook

# COMMAND ----------

enable_splink(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Getting the data (again)
# MAGIC 
# MAGIC And retrieve the data we should already be familiar with.

# COMMAND ----------

# DBTITLE 1,FIX AFTER WE AGREE ON DATA LOCATION
# data = spark.read.table(f"splink_{db_name}.companies_with_officers")
table_name = "companies_with_officers"
data = spark.read.format("delta").load(f"dbfs:/Filestore/Users/{username}/{table_name}")

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
  "em_convergence": 0.01,
  "blocking_rules_to_generate_predictions": blocking_rules_to_generate_predictions
}


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
  model = SplinkLinkerModel()
  model.spark_linker(data)
  model.set_settings(settings)
  model.set_deterministic_rules(deterministic_rules)
  model.set_blocking_rules(blocking_rules_to_generate_predictions)
  model.set_comparisons(comparisons)
  model.set_target_rows(1e7)
  model.set_stage1_columns(["name", "date_of_birth"])
  model.set_stage2_columns(["surname", "forenames"])
  model.fit(data)

  # Make predictions on the data
  predictions = model.get_linker().predict().as_pandas_dataframe()

  # Log model settings as JSON to MLflow run
  model.log_settings_as_json("linker_settings.json")

  # Log parameters to MLflow
  params = get_hyperparameters(model.get_settings_object())
  all_comparisons = get_all_comparisons(model.get_settings_object())
  charts = get_linker_charts(model.get_linker(), True, True)

  mlflow.log_params(params)
  for _ in all_comparisons:
      mlflow.log_params(_)
  mlflow.log_dict(model.get_settings_object(), "linker.json")

  # Log model to MLflow
  mlflow.pyfunc.log_model("linker", python_model=model)

  # Log Charts to MLflow
  for name, chart in charts.items():
      model._log_chart(name, chart)

  # Evaluate linker model and log loss as metric
  loss, fig = get_match_probabilty_loss(predictions)
  mlflow.log_metric("match_probability_loss", loss)
  mlflow.log_figure(fig, 'prediction_distribution_loss.png')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using the trained model
# MAGIC 
# MAGIC In the above MLflow run, we created and trained a `model` object. We can use this object directly for further predictions, or to make use of Splink's many built-in exploratory tools.

# COMMAND ----------

baseline_linker = model.linker
baseline_linker.m_u_parameters_chart()

# COMMAND ----------

# MAGIC %md
# MAGIC As above, we can use this `loaded_linker` to use Splink's functionality:

# COMMAND ----------

df_clusters = baseline_linker.cluster_pairwise_predictions_at_threshold(baseline_linker.predict(), 0.9)
df_predictions = baseline_linker.predict()
baseline_linker.cluster_studio_dashboard(df_predictions, df_clusters, "cluster_studio.html", sampling_method="by_cluster_size", overwrite=True)

IFrame(src="./cluster_studio.html", width="100%", height=1200)

# COMMAND ----------

# MAGIC %md
# MAGIC You can also call the `get_match_probabilty_loss` method directly on the predictions (as long as it's a Pandas dataframe) at any point to check your predictions.

# COMMAND ----------

baseline_loss, baseline_fig = get_match_probabilty_loss(df_predictions.as_pandas_dataframe())

baseline_loss

# COMMAND ----------

# MAGIC %md
# MAGIC To visualise clusters together, you can use the `visualise_linked_records` for a static (but less readable) picture of the connected records.
# MAGIC 
# MAGIC Use the `linkage_threshold` argument to set the minimum match probability, and the `min_linked_records` argument to set a minimum number of records in a cluster for it to be visualised. Avoid setting these too low or the visualisation will be unreadable.

# COMMAND ----------

visualise_linked_records(predictions, min_linked_records=10, linkage_threshold=0.9)

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
